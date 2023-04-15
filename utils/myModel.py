import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.networks.blocks.convolutions import Convolution

class MyModel(nn.Module):
    def __init__(self,img_size=(64, 64, 64)):
        super().__init__()
        self.swinUNETR = SwinUNETR(
                                img_size=img_size,
                                in_channels=1,
                                out_channels=1,
                                feature_size=12,
                                use_checkpoint=True)
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
        # self.conv1 = Convolution(spatial_dims=3, in_channels=14, out_channels=1, kernel_size=1)
        # self.conv2 = Convolution(spatial_dims=3, in_channels=1, out_channels=1, kernel_size=(1, 1, 64), strides=1, padding=0, act="sigmoid")

    
    def forward(self, x):
        # if x[0].size() != (1, 64, 64, 64):
        #     print(x.size())
        #     raise ValueError("Input size is not correct")
        x_out = self.swinUNETR(x)
        # x_out = self.conv1(x_out)
        # x_out = self.conv2(x_out)
        return x_out
    
    def load_swin_ckpt(self, model_dict, strict: bool = True):
        self.swinUNETR.load_state_dict(model_dict, strict)
        pass