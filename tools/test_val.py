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
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping
from monai.config.type_definitions import NdarrayOrTensor

print(monai.__version__)
torch.cuda.set_device(0)

class Copy(Transform):
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __call__(self, data):
        assert isinstance(data, torch.Tensor)
        data = data.repeat(1, self.num_channel, 1, 1)  # output = (batch_size=1, num_channel, H, W)
        return data
    
class Copyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection, num_channel) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, )
        self.adder = Copy(num_channel)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label", 'inklabels'], reader="NumpyReader"),
        AddChanneld(keys=["image"]),
        Copyd(keys=["label", 'inklabels'], num_channel=65), 
        Orientationd(keys=["image", "label", 'inklabels'], axcodes="RAS"),
        # Spacingd(
        #     keys=["image", "label", 'inklabels'], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
        # ),
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=65535.0, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", 'inklabels'], source_key="image"),
        ToTensord(keys=["image", 'inklabels']),
    ]
)



# (batch_size, in_channel, H, W, D)
data = torch.ones(7, 1, 64, 64, 64)

model = MyModel().cuda(0)

model.eval()

torch.cuda.empty_cache()

pred = model(data.cuda(0))

print(pred.size())