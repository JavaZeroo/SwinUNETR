import torch.nn as nn
from monai.networks.nets import SwinUNETR, UNet, FlexibleUNet, BasicUNetPlusPlus
from monai.networks.blocks.convolutions import Convolution


encoder_feature_channel = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SwinUNETR(
            img_size=(96,96,96),
            in_channels=1,
            out_channels=14,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
        # self.conv1 = Convolution(spatial_dims=3, in_channels=14, out_channels=1, kernel_size=1)
        self.conv2 = Convolution(spatial_dims=3, in_channels=1, out_channels=1, kernel_size=(1, 1, 64), strides=1, padding=0, act="sigmoid")

    
    def forward(self, x):
        if x[0].size() != (1, 64, 64, 64):
            print(x.size())
            raise ValueError("Input size is not correct")
        x_out = self.swinUNETR(x)
        # x_out = self.conv1(x_out)
        x_out = self.conv2(x_out)
        return x_out
    
    def load_swin_ckpt(self, model_dict, strict: bool = True):
        self.swinUNETR.load_state_dict(model_dict, strict)
        pass
    
class MyModel2d(nn.Module):
    def __init__(self,img_size=(192, 192)):
        super().__init__()
        self.swinUNETR = SwinUNETR(
                                img_size=img_size,
                                in_channels=65,
                                out_channels=1,
                                feature_size=12,
                                use_checkpoint=True, 
                                spatial_dims=2
                                )

    
    def forward(self, x):
        x_out = self.swinUNETR(x)
        return x_out
    
    def load_swin_ckpt(self, model_dict, strict: bool = True):
        self.swinUNETR.load_state_dict(model_dict, strict)
        pass
    
class MyModel3dunet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.conv1 = Convolution(spatial_dims=3, in_channels=1, out_channels=1, kernel_size=(1, 1, 64), strides=1, padding=0, act="sigmoid")
    
    def forward(self, x):
        x_out = self.unet(x)
        x_out = self.conv1(x_out)
        return x_out
    

class MyFlexibleUNet2d(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.flexibleUNet = FlexibleUNet(
                in_channels=args.num_channel,
                out_channels=1,
                backbone=f"efficientnet-{args.eff}",
                pretrained=True,
                spatial_dims=2,
                dropout=0.0,
            )
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x_out = self.flexibleUNet(x)
        x_out = self.sig(x_out)
        return x_out
        
        

class ConvLSTM(nn.Module):
    def __init__(self, in_channels=320, out_channels=320, kernel_size=1, padding=0, batch_first=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.lstm = nn.LSTM(256, 256, batch_first=batch_first)

    def forward(self, x):

        last_feature = x[-1]

        batch_size, channels, height, width = last_feature.shape

        # Apply 2D convolution
        last_feature = self.conv(last_feature)

        # Reshape output for LSTM
        last_feature = last_feature.view(batch_size, -1, height * width)

        # Pass through LSTM
        lstm_out, _ = self.lstm(last_feature)

        # Reshape output back to original shape
        last_feature = lstm_out.view(batch_size, channels, height, width)

        x[-1] = last_feature

        return x


class MyFlexibleUNet2dLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.flexibleUNet = FlexibleUNet(
            in_channels=args.num_channel,
            out_channels=1,
            backbone=f"efficientnet-{args.eff}",
            pretrained=True,
            spatial_dims=2,
            dropout=0.0,
        )
        # Add ConvLSTM layer after the last convolution layer in the encoder
        assert args.roi_x == args.roi_y, "ROI x and y must be the same"
        self.debug = args.debug
        channels = encoder_feature_channel[f"efficientnet-{args.eff}"]
        self.conv_lstm = ConvLSTM(in_channels=channels[-1], out_channels=channels[-1], kernel_size=1, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        if self.debug:
            print(x.shape)
        x_out = self.flexibleUNet.encoder(x)
        if self.debug:
            for i in x_out:
                print(i.shape)
        x_out = self.conv_lstm(x_out)
        x_out = self.flexibleUNet.decoder(x_out)
        x_out = self.flexibleUNet.segmentation_head(x_out)
        x_out = self.sig(x_out)
        return x_out
    
class MyBasicUNetPlusPlus(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.basicUNetPlusPlus = BasicUNetPlusPlus(in_channels=1, out_channels=1)
        
    def forward(self, x):
        x_out = self.basicUNetPlusPlus(x)
        return x_out

class ConvLSTM_block(nn.Module):
    def __init__(self, lstm_length, in_channels=320, out_channels=320, kernel_size=1, padding=0, batch_first=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.lstm = nn.LSTM(lstm_length, lstm_length, batch_first=batch_first)

    def forward(self, x):


        batch_size, channels, height, width = x.shape

        # Apply 2D convolution
        x_out = self.conv(x)

        # Reshape output for LSTM
        x_out = x_out.view(batch_size, -1, height * width)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x_out)

        # Reshape output back to original shape
        x_out = lstm_out.view(batch_size, channels, height, width)

        return x_out

class MultiScaleConvLSTM(nn.Module):
    def __init__(self, args, backbone_channels, kernel_size=1, padding=0, batch_first=True):
        super().__init__()
        assert args.roi_x == args.roi_y, "ROI x and y must be the same"
        conv_list = []
        for i, channel in enumerate(backbone_channels):
            if i == 0 or i ==1:
                conv_list.append(None)
                continue
            lstm_length = int((args.roi_x / (2 ** (i+1)))**2)
            conv_list.append(ConvLSTM_block(lstm_length, channel, channel, kernel_size, padding, batch_first))
        self.convlstm_layers = nn.ModuleList(conv_list)

    def forward(self, features_list):
        for i, convlstm in enumerate(self.convlstm_layers):
            if convlstm is None:
                continue
            features_list[i] = convlstm(features_list[i])
        return features_list

class MyFlexibleUNet2dMultiScaleLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.flexibleUNet = FlexibleUNet(
            in_channels=args.num_channel,
            out_channels=1,
            backbone=f"efficientnet-{args.eff}",
            pretrained=True,
            spatial_dims=2,
            dropout=args.dropout_rate,
        )
        # Add MultiScaleConvLSTM layer after the last convolution layer in the encoder
        backbone_channels = encoder_feature_channel[f"efficientnet-{args.eff}"]
        self.multi_scale_conv_lstm = MultiScaleConvLSTM(args=args, backbone_channels=backbone_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_out = self.flexibleUNet.encoder(x)
        x_out = self.multi_scale_conv_lstm(x_out)
        x_out = self.flexibleUNet.decoder(x_out)
        x_out = self.flexibleUNet.segmentation_head(x_out)
        x_out = self.sig(x_out)
        return x_out



class MultiScaleConvLSTM3d(nn.Module):
    def __init__(self, args, backbone_channels, kernel_size=1, padding=0, batch_first=True):
        super().__init__()
        assert args.roi_x == args.roi_y, "ROI x and y must be the same"
        conv_list = []
        for i, channel in enumerate(backbone_channels):
            if i == 0 or i ==1:
                conv_list.append(None)
                continue
            lstm_length = int((args.roi_x / (2 ** (i+1)))**2)
            conv_list.append(ConvLSTM_block(lstm_length, channel, channel, kernel_size, padding, batch_first))
        self.convlstm_layers = nn.ModuleList(conv_list)

    def forward(self, features_list):
        for i, convlstm in enumerate(self.convlstm_layers):
            if convlstm is None:
                continue
            features_list[i] = convlstm(features_list[i])
        return features_list

class MyFlexibleUNet3dMultiScaleLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.flexibleUNet = FlexibleUNet(
            in_channels=1,
            out_channels=1,
            backbone=f"efficientnet-{args.eff}",
            spatial_dims=3,
            dropout=args.dropout_rate,
        )
        # Add MultiScaleConvLSTM layer after the last convolution layer in the encoder
        backbone_channels = encoder_feature_channel[f"efficientnet-{args.eff}"]
        self.multi_scale_conv_lstm = MultiScaleConvLSTM(args=args, backbone_channels=backbone_channels)
        self.conv1 = Convolution(spatial_dims=3, in_channels=1, out_channels=1, kernel_size=(1, 1, args.num_channel), strides=1, padding=0)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_out = self.flexibleUNet.encoder(x)
        # x_out = self.multi_scale_conv_lstm(x_out)
        x_out = self.flexibleUNet.decoder(x_out)
        x_out = self.flexibleUNet.segmentation_head(x_out)
        x_out = self.conv1(x_out)
        x_out = self.sig(x_out)
        return x_out
    
    
################## NEW MODEL ##################
from timm.models.resnet import resnet10t, resnet34d
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from einops import rearrange
import torch.nn.functional as F
import torch

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
		v = batch['volume']
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

		output = {
			'ink' : torch.sigmoid(logit2),
		}
		return output

################## NEW MODEL ##################
