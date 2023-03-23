import torch.nn as nn
from monai.networks.nets import SwinUNETR

class MyModel(nn.Module):
    def __init__(self,img_size=(64, 64, 64)):
        super().__init__()
        self.swinUNETR = SwinUNETR(
                                img_size=img_size,
                                in_channels=1,
                                out_channels=1,
                                feature_size=12,
                                use_checkpoint=True)
        self.conv = nn.Conv2d(64, 1, 1, 1)
    
    def forward(self, x):
        x_out = self.swinUNETR(x)[0]
        print(x_out.size())
        x_out = self.conv(x_out)
        print(x_out.size())
        return x_out
    
    def load_swin_ckpt(self, model_dict, strict: bool = True):
        self.swinUNETR.load_state_dict(model_dict, strict)
        pass