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
files = [
    "train/1/surface_volume/00.tif",
    "train/1/surface_volume/01.tif",
    "train/1/surface_volume/02.tif",
    "train/1/surface_volume/03.tif",
    "train/1/surface_volume/04.tif",
    "train/1/surface_volume/05.tif",
    "train/1/surface_volume/06.tif",
    "train/1/surface_volume/07.tif",
    "train/1/surface_volume/08.tif",
    "train/1/surface_volume/09.tif",
    "train/1/surface_volume/10.tif",
    "train/1/surface_volume/11.tif",
    "train/1/surface_volume/12.tif",
    "train/1/surface_volume/13.tif",
    "train/1/surface_volume/14.tif",
    "train/1/surface_volume/15.tif",
    "train/1/surface_volume/16.tif",
    "train/1/surface_volume/17.tif",
    "train/1/surface_volume/18.tif",
    "train/1/surface_volume/19.tif",
    "train/1/surface_volume/20.tif",
    "train/1/surface_volume/21.tif",
    "train/1/surface_volume/22.tif",
    "train/1/surface_volume/23.tif",
    "train/1/surface_volume/24.tif",
    "train/1/surface_volume/25.tif",
    "train/1/surface_volume/26.tif",
    "train/1/surface_volume/27.tif",
    "train/1/surface_volume/28.tif",
    "train/1/surface_volume/29.tif",
    "train/1/surface_volume/30.tif",
    "train/1/surface_volume/31.tif",
    "train/1/surface_volume/32.tif",
    "train/1/surface_volume/33.tif",
    "train/1/surface_volume/34.tif",
    "train/1/surface_volume/35.tif",
    "train/1/surface_volume/36.tif",
    "train/1/surface_volume/37.tif",
    "train/1/surface_volume/38.tif",
    "train/1/surface_volume/39.tif",
    "train/1/surface_volume/40.tif",
    "train/1/surface_volume/41.tif",
    "train/1/surface_volume/42.tif",
    "train/1/surface_volume/43.tif",
    "train/1/surface_volume/44.tif",
    "train/1/surface_volume/45.tif",
    "train/1/surface_volume/46.tif",
    "train/1/surface_volume/47.tif",
    "train/1/surface_volume/48.tif",
    "train/1/surface_volume/49.tif",
    "train/1/surface_volume/50.tif",
    "train/1/surface_volume/51.tif",
    "train/1/surface_volume/52.tif",
    "train/1/surface_volume/53.tif",
    "train/1/surface_volume/54.tif",
    "train/1/surface_volume/55.tif",
    "train/1/surface_volume/56.tif",
    "train/1/surface_volume/57.tif",
    "train/1/surface_volume/58.tif",
    "train/1/surface_volume/59.tif",
    "train/1/surface_volume/60.tif",
    "train/1/surface_volume/61.tif",
    "train/1/surface_volume/62.tif",
    "train/1/surface_volume/63.tif",
    "train/1/surface_volume/64.tif"
]
ROOT = Path(r'/root/autodl-tmp/vesuvius-challenge-ink-detection/')
files = [str(ROOT / f) for f in files]
from monai.data import PILReader
train_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"], reader='PILReader')
    ]
)
from monai.data import load_decathlon_datalist
import os
datalist_json = os.path.join("/root/autodl-tmp/vesuvius-challenge-ink-detection", "first.json")
seg, _ = LoadImage(image_only=False, reader='PILReader')(ROOT / 'train' / '1'/ 'mask.png')
print(seg.shape)

datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir="/root/autodl-tmp/vesuvius-challenge-ink-detection")
mydict = {"image":files,'label': '/root/autodl-tmp/vesuvius-challenge-ink-detection/train/3/mask.png'}
print(mydict)
img = apply_transform(train_transform, mydict)
print(img['image'].shape)
# for i in img:
#     print(i.image)