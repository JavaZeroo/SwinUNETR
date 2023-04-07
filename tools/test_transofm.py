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
from monai import transforms
print(monai.__version__)
torch.cuda.set_device(0)
images = [
    "train/1/imgs/00/00_15.npy",
    "train/1/imgs/01/01_15.npy",
    "train/1/imgs/02/02_15.npy",
    "train/1/imgs/03/03_15.npy",
    "train/1/imgs/04/04_15.npy",
    "train/1/imgs/05/05_15.npy",
    "train/1/imgs/06/06_15.npy",
    "train/1/imgs/07/07_15.npy",
    "train/1/imgs/08/08_15.npy",
    "train/1/imgs/09/09_15.npy",
    "train/1/imgs/10/10_15.npy",
    "train/1/imgs/11/11_15.npy",
    "train/1/imgs/12/12_15.npy",
    "train/1/imgs/13/13_15.npy",
    "train/1/imgs/14/14_15.npy",
    "train/1/imgs/15/15_15.npy",
    "train/1/imgs/16/16_15.npy",
    "train/1/imgs/17/17_15.npy",
    "train/1/imgs/18/18_15.npy",
    "train/1/imgs/19/19_15.npy",
    "train/1/imgs/20/20_15.npy",
    "train/1/imgs/21/21_15.npy",
    "train/1/imgs/22/22_15.npy",
    "train/1/imgs/23/23_15.npy",
    "train/1/imgs/24/24_15.npy",
    "train/1/imgs/25/25_15.npy",
    "train/1/imgs/26/26_15.npy",
    "train/1/imgs/27/27_15.npy",
    "train/1/imgs/28/28_15.npy",
    "train/1/imgs/29/29_15.npy",
    "train/1/imgs/30/30_15.npy",
    "train/1/imgs/31/31_15.npy",
    "train/1/imgs/32/32_15.npy",
    "train/1/imgs/33/33_15.npy",
    "train/1/imgs/34/34_15.npy",
    "train/1/imgs/35/35_15.npy",
    "train/1/imgs/36/36_15.npy",
    "train/1/imgs/37/37_15.npy",
    "train/1/imgs/38/38_15.npy",
    "train/1/imgs/39/39_15.npy",
    "train/1/imgs/40/40_15.npy",
    "train/1/imgs/41/41_15.npy",
    "train/1/imgs/42/42_15.npy",
    "train/1/imgs/43/43_15.npy",
    "train/1/imgs/44/44_15.npy",
    "train/1/imgs/45/45_15.npy",
    "train/1/imgs/46/46_15.npy",
    "train/1/imgs/47/47_15.npy",
    "train/1/imgs/48/48_15.npy",
    "train/1/imgs/49/49_15.npy",
    "train/1/imgs/50/50_15.npy",
    "train/1/imgs/51/51_15.npy",
    "train/1/imgs/52/52_15.npy",
    "train/1/imgs/53/53_15.npy",
    "train/1/imgs/54/54_15.npy",
    "train/1/imgs/55/55_15.npy",
    "train/1/imgs/56/56_15.npy",
    "train/1/imgs/57/57_15.npy",
    "train/1/imgs/58/58_15.npy",
    "train/1/imgs/59/59_15.npy",
    "train/1/imgs/60/60_15.npy",
    "train/1/imgs/61/61_15.npy",
    "train/1/imgs/62/62_15.npy",
    "train/1/imgs/63/63_15.npy",
    "train/1/imgs/64/64_15.npy"
]
ROOT = Path('/mnt/e/Code/ink_data')
files = [str(ROOT / f) for f in images]
from monai.data import PILReader


import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist

from monai.transforms.transform import Transform
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping
from monai.config.type_definitions import NdarrayOrTensor

class Copy(Transform):
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __call__(self, data):
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

train_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label", 'inklabels']), 
        transforms.AddChanneld(keys=["image"]),
        Copyd(keys=["label", 'inklabels'], num_channel=65), 
        transforms.Orientationd(keys=["image", "label", 'inklabels'], axcodes="RAS"),
        # transforms.Spacingd(
        #     keys=["image", "label", 'inklabels'], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
        # ),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=65535.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.CropForegroundd(keys=["image", "label", 'inklabels'], source_key="image"),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=0,
            num_samples=2,
            image_key="image",
            image_threshold=0,
            allow_smaller=False,
        ),
    ]
)
from monai.data import load_decathlon_datalist
# import os
# datalist_json = os.path.join("/root/autodl-tmp/vesuvius-challenge-ink-detection", "first.json")
seg, _ = LoadImage(image_only=False)('/mnt/e/Code/ink_data/train/1/mask/mask_15.npy')
print(seg.shape)

datalist = load_decathlon_datalist("thrid.json", True, "training", base_dir="/mnt/e/Code/ink_data")
# mydict = {"image":files,'label': '/mnt/e/Code/ink_data/train/1/mask/mask_15.npy', "inklabels": "'/mnt/e/Code/ink_data/train/1/ink_label/ink_label_15.npy"}
# print(mydict)
fails = {"all":[]}
print(datalist[0]['label'].split('/')[-3])
print(datalist[0]['label'].split('/')[-1].split('_')[-1])
from tqdm import tqdm
for i in tqdm(range(len(datalist))):
    try:
        img = apply_transform(train_transform, datalist[i])
    except:
        add_line = {"img":str(datalist[i]['label'].split('/')[-3]), 'nums':datalist[i]['label'].split('/')[-1].split('_')[-1]}
        fails['all'].append(add_line)
        print(add_line)
        continue
    for j in range(len(img)):
        # check size if it is 1x64x64x64
        try:
            if img[j]['image'].shape != (1, 64, 64, 64) or img[j]['label'].shape != (1, 64, 64, 64):
                print(img[j]['image'].shape, img[j]['label'].shape, img[j]['inklabels'].shape)
        except:
            print(img)
            print(datalist[i]['label'])
            print("==================Attention!!!==================")
            continue
import json           
with open("fails.json", "w") as f:
    json.dump(fails, f)