import sys
sys.path.append('..')
sys.path.append('.')
# import os
# import shutil
# import SimpleITK as sitk  # noqa: N813
import numpy as np
# import itk
# import tempfile
import monai
# from monai.data import PILReader
from monai.transforms import *
# from monai.config import print_config
from pathlib import Path
from monai.networks.nets import SwinUNETR
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from utils.myModel import MyModel, MyModel3dunet, MyFlexibleUNet2dMultiScaleLSTM, MyFlexibleUNet3dMultiScaleLSTM
from easydict import EasyDict as edict
from timm.models.resnet import resnet10t, resnet34d
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from einops import rearrange
import torch.nn.functional as F
# print(monai.__version__)
# torch.cuda.set_device(0)
# args = edict(roi_x=128, roi_y=128, num_channel=40, eff='b5', dropout_rate=0.0)
# # (batch_size, in_channel, H, W, D)
# data = torch.ones(8,1,  args.roi_x, args.roi_y, args.num_channel).cuda()

# model = MyFlexibleUNet3dMultiScaleLSTM(args=args).cuda()

# model.eval()

# torch.cuda.empty_cache()

# pred = model(data)

# print(pred.cpu().size())




class Config(object):
	mode = [
		#'train', #
		'test', 'skip_fake_test',
	]
	crop_fade  = 56
	crop_size  = 384
	crop_depth = 5
	infer_fragment_z = [28, 37]

CFG = Config()
CFG.is_tta = True #True

class SmpUnetDecoder(nn.Module):
	def __init__(self,
	         in_channel,
	         skip_channel,
	         out_channel,
	    ):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel,]+ out_channel[:-1]
		s_channel = skip_channel
		o_channel = out_channel
		block = [
			DecoderBlock(i, s, o, use_batchnorm=True, attention_type=None)
			for i, s, o in zip(i_channel, s_channel, o_channel)
		]
		self.block = nn.ModuleList(block)

	def forward(self, feature, skip):
		d = self.center(feature)
		decode = []
		for i, block in enumerate(self.block):
			s = skip[i]
			d = block(d, s)
			decode.append(d)

		last  = d
		return last, decode

class Net(nn.Module):
	def __init__(self,):
		super().__init__()
		self.output_type = ['inference', 'loss']

		conv_dim = 64
		encoder1_dim  = [conv_dim, 64, 128, 256, 512, ]
		decoder1_dim  = [256, 128, 64, 64,]

		self.encoder1 = resnet34d(pretrained=False, in_chans=CFG.crop_depth)

		self.decoder1 = SmpUnetDecoder(
			in_channel   = encoder1_dim[-1],
			skip_channel = encoder1_dim[:-1][::-1],
			out_channel  = decoder1_dim,
		)
		# -- pool attention weight
		self.weight1 = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
			) for dim in encoder1_dim
		])
		self.logit1 = nn.Conv2d(decoder1_dim[-1],1,kernel_size=1)

		#--------------------------------
		#
		encoder2_dim  = [64, 128, 256, 512]#
		decoder2_dim  = [128, 64, 32, ]
		self.encoder2 = resnet10t(pretrained=False, in_chans=decoder1_dim[-1])

		self.decoder2 = SmpUnetDecoder(
			in_channel   = encoder2_dim[-1],
			skip_channel = encoder2_dim[:-1][::-1],
			out_channel  = decoder2_dim,
		)
		self.logit2 = nn.Conv2d(decoder2_dim[-1],1,kernel_size=1)

	def forward(self, batch):
		v = batch
		B,C,H,W = v.shape
		vv = [
			v[:,i:i+CFG.crop_depth] for i in [0,2,4,]
		]
		K = len(vv)
		x = torch.cat(vv,0)
		#x = v

		#----------------------
		encoder = []
		e = self.encoder1
		x = e.conv1(x)
		x = e.bn1(x)
		x = e.act1(x);
		encoder.append(x)
		x = F.avg_pool2d(x, kernel_size=2, stride=2)
		x = e.layer1(x);
		encoder.append(x)
		x = e.layer2(x);
		encoder.append(x)
		x = e.layer3(x);
		encoder.append(x)
		x = e.layer4(x);
		encoder.append(x)
		# print('encoder', [f.shape for f in encoder])

		for i in range(len(encoder)):
			e = encoder[i]
			f = self.weight1[i](e)
			_, c, h, w = e.shape
			f = rearrange(f, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w)  #
			e = rearrange(e, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w)  #
			w = F.softmax(f, 1)
			e = (w * e).sum(1)
			encoder[i] = e

		feature = encoder[-1]
		skip = encoder[:-1][::-1]
		last, decoder = self.decoder1(feature, skip)
		logit1 = self.logit1(last)

		#----------------------
		x = last #.detach()
		#x = F.avg_pool2d(x,kernel_size=2,stride=2)
		encoder = []
		e = self.encoder2
		x = e.layer1(x); encoder.append(x)
		x = e.layer2(x); encoder.append(x)
		x = e.layer3(x); encoder.append(x)
		x = e.layer4(x); encoder.append(x)

		feature = encoder[-1]
		skip = encoder[:-1][::-1]
		last, decoder = self.decoder2(feature, skip)
		logit2 = self.logit2(last)
		logit2 = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False, antialias=True)

		return torch.sigmoid(logit2)

def run_check_net():

    height,width =  CFG.crop_size, CFG.crop_size
    depth = CFG.infer_fragment_z[1]-CFG.infer_fragment_z[0]
    batch_size = 3

    batch = torch.from_numpy( np.random.choice(256, (batch_size, depth, height, width))).cuda().float()
    net = Net().cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)
            
    print('batch')
    print(f'{batch.shape}')

    print('output')
    print(f'{output.shape}')


run_check_net()
print('net ok !!!')