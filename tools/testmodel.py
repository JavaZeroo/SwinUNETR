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
from utils.myModel import MyModel, MyModel3dunet, MyFlexibleUNet2dMultiScaleLSTM
from easydict import EasyDict as edict
print(monai.__version__)
# torch.cuda.set_device(0)
args = edict(roi_x=512, roi_y=512, num_channel=65, eff='b5')
# (batch_size, in_channel, H, W, D)
data = torch.ones(8, 65, args.roi_x, args.roi_y).cuda()

model = MyFlexibleUNet2dMultiScaleLSTM(args=args).cuda()

model.eval()

torch.cuda.empty_cache()

pred = model(data)

print(pred.cpu().size())