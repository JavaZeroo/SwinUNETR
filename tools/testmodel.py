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
from utils.myModel import MyModel

print(monai.__version__)
torch.cuda.set_device(0)

# (batch_size, in_channel, H, W, D)
data = torch.ones(7, 1, 64, 64, 64)

model = MyModel().cuda(0)

model.eval()

torch.cuda.empty_cache()

pred = model(data.cuda(0))

print(pred.size())