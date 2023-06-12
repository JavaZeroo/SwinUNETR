import torch.nn as nn
from monai.networks.nets import SwinUNETR, UNet, FlexibleUNet, BasicUNetPlusPlus
from monai.networks.blocks.convolutions import Convolution
import segmentation_models_pytorch as smp

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
    
    
class MyBasicUNetPlusPlus2d(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.basicUNetPlusPlus = smp.UnetPlusPlus(
            encoder_name='resnext50_32x4d', 
            encoder_weights=None, 
            classes=1, 
            in_channels=args.num_channel,
            activation="sigmoid")
        # self.basicUNetPlusPlus = BasicUNetPlusPlus(spatial_dims=2, in_channels=args.num_channel, out_channels=1)
        # self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x_out = self.basicUNetPlusPlus(x)
        # print(x_out.shape)
        # x_out = self.sig(x)
        # print(x_out.shape)
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



##### transformer

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            dim_feedforward=embed_dim*forward_expansion
        )

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        add_attention = query + attention
        norm_attention = self.norm1(add_attention)
        output = self.transformer_block(norm_attention)
        return output

class ConvLSTMtransformer_block(nn.Module):
    def __init__(self, lstm_length, in_channels=320, out_channels=320, kernel_size=1, padding=0, batch_first=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # self.lstm = nn.LSTM(lstm_length, lstm_length, batch_first=batch_first)
        self.transformer_block = TransformerBlock(embed_dim=lstm_length, num_heads=4, dropout=0.1, forward_expansion=4)
    def forward(self, x):


        batch_size, channels, height, width = x.shape

        # Apply 2D convolution
        x_out = self.conv(x)

        # Reshape output for LSTM
        x_out = x_out.view(batch_size, -1, height * width)

        # Pass through LSTM
        lstm_out = self.transformer_block(x_out, x_out, x_out)
                
        # Reshape output back to original shape
        x_out = lstm_out.view(batch_size, channels, height, width)

        return x_out
    
# 在MultiScaleConvLSTM类中添加TransformerBlock
class MultiScaleConvLSTMtransformer(nn.Module):
    def __init__(self, args, backbone_channels, kernel_size=1, padding=0, batch_first=True):
        super().__init__()
        assert args.roi_x == args.roi_y, "ROI x and y must be the same"
        conv_list = []
        for i, channel in enumerate(backbone_channels):
            if i == 0 or i ==1 or i ==2:
                conv_list.append(None)
                continue
            lstm_length = int((args.roi_x / (2 ** (i+1)))**2)
            conv_list.append(ConvLSTMtransformer_block(lstm_length, channel, channel, kernel_size, padding, batch_first))
        self.convlstm_layers = nn.ModuleList(conv_list)

    def forward(self, features_list):
        for i, convlstm in enumerate(self.convlstm_layers):
            if convlstm is None:
                continue
            features_list[i] = convlstm(features_list[i])
        # take the last element in the feature list as query, key, and value for the attention
        # query = key = value = features_list[-1]
        # # pass the output of the convlstm_layers to the TransformerBlock
        # output = self.transformer_block(value, key, query)
        return features_list

#####


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
        self.multi_scale_conv_lstm = MultiScaleConvLSTMtransformer(args=args, backbone_channels=backbone_channels)
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

# class Config(object):
#     def __init__(self, args) -> None:
#         self.mode = [
#             #'train', #
#             'test', 'skip_fake_test',
#         ]
#         self.crop_fade  = 56
#         self.crop_size  = args.roi_x
#         self.crop_depth = 5
#         self.infer_fragment_z = args.z_range
#         pass



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

class Config(object):
    valid_threshold = 0.80
    beta = 1
    crop_fade  = 32
    crop_size  = 128 #256 
    crop_depth = 5
    infer_fragment_z = [
        32-16,
        32+16,
    ]#32 slices
    dz = 0
CFG1 = Config()


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.output_type = ['inference', 'loss']

        # --------------------------------
        CFG = CFG1
        self.crop_depth = CFG.crop_depth

        conv_dim = 64
        encoder_dim = [conv_dim, 64, 128, 256, 512, ]
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = resnet34d(pretrained=True, in_chans=self.crop_depth)

        self.decoder = SmpUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
        )
        self.logit = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)

        # --------------------------------
        self.aux = nn.ModuleList([
            nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
        ])


    def forward(self, batch):
        v = batch
        B, C, H, W = v.shape
        vv = [
            v[:, i:i + self.crop_depth] for i in range(0,C-self.crop_depth+1,2)
        ]
        K = len(vv)
        x = torch.cat(vv, 0)

        # ---------------------------------

        encoder = []
        e = self.encoder

        x = e.conv1(x)
        x = e.bn1(x)
        x = e.act1(x); encoder.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = e.layer1(x); encoder.append(x)
        x = e.layer2(x); encoder.append(x)
        x = e.layer3(x); encoder.append(x)
        x = e.layer4(x); encoder.append(x)
        ##[print('encoder',i,f.shape) for i,f in enumerate(encoder)]

        for i in range(len(encoder)):
            e = encoder[i]
            _, c, h, w = e.shape
            e = rearrange(e, '(K B) c h w -> K B c h w', K=K, B=B, h=h, w=w)
            encoder[i] = e.mean(0)

        last, decoder = self.decoder(feature = encoder[-1], skip = encoder[:-1][::-1]  + [None])


        # ---------------------------------
        logit = self.logit(last)

        if 1:
            if logit.shape[2:]!=(H, W):
                logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
            output = torch.sigmoid(logit)

        return output

################## NEW MODEL ##################


