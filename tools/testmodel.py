
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
    # "train/1/surface_volume/64.tif"
]
ROOT = Path(r'/mnt/e/Code/vesuvius-challenge-ink-detection')
files = [ROOT / f for f in files]

label = ROOT / 'train' / '1'/ 'mask.png'
labels = [label for i in range(len(files))]
print("Loading data...")
# raw, meta = LoadImage(image_only=False, reader='PILReader')(files)
raw = torch.rand(65, 1000, 1500)
seg = torch.rand(65, 1000, 1500)
# seg, _ = LoadImage(image_only=False, reader='PILReader')(labels)
print("Finished Loading data...")
print(f"image data shape: {raw.shape}")
print(f"image data shape: {seg.shape}")
# https://blog.csdn.net/u014264373/article/details/113742194
raw = AddChannel()(raw)
seg = AddChannel()(seg)
print(f"image data shape: {raw.shape}")
print(f"image data shape: {seg.shape}")
raw = Orientation(axcodes="RAS")(raw)
seg = Orientation(axcodes="RAS")(seg)
print(f"Orientation: {raw.shape}")
raw = Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear")(raw)
seg = Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear")(seg)
print(f"Spacing: {raw.shape}")
raw = ScaleIntensityRange(
    a_min=0, a_max=60000, b_min=0.0, b_max=1.0, clip=True
)(raw)
print(f"ScaleIntensityRange: {raw.shape}")
raw = CropForeground( source_key="image")(raw)
seg = CropForeground( source_key="image")(seg)
print(f"CropForeground: {raw.shape}")
print("==================")
raw = RandCropByPosNegLabel(
    label=None,
    spatial_size=(64, 64, 64),
    pos=1,
    neg=1,
    num_samples=4,
    image_threshold=0,
)(raw, label=seg)
print(f"RandCropByPosNegLabel: {raw[0].shape}")
print(type(raw))
raw = RandFlip( prob=0.2,  spatial_axis=0)(raw)
print(f"RandFlip: {raw[0].shape}")
raw = RandFlip( prob=0.2, spatial_axis=1)(raw)
print(f"RandFlip: {raw[0].shape}")
raw = RandFlip( prob=0.2, spatial_axis=2)(raw)
print(f"RandFlip: {raw[0].shape}")
raw = RandRotate90( prob=0.2, max_k=3)(raw)
print(f"RandRotate90: {raw[0].shape}")
raw = RandScaleIntensity(factors=0.1, prob=0.1)(raw)
print(f"RandScaleIntensity: {raw[0].shape}")
raw = RandShiftIntensity(offsets=0.1, prob=0.1)(raw)
print(f"RandShiftIntensity: {raw[0].shape}")
raw = ToTensor()(raw)
print(f"{type(raw)}")
raw = torch.Tensor(raw).cuda(0)
print(f"{type(raw)}")



# raw_croped = raw[:,1000:1100,1000:1100]
# data = torch.Tensor(raw_croped).view(1, raw_croped.shape[0], raw_croped.shape[1], raw_croped.shape[2]).cuda(0)

# (batch_size, in_channel, H, W, D)
data = torch.rand((1, 1, 96, 96, 96)).cuda(0)
data.size()



model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=12,
    use_checkpoint=True,
).cuda(0)

model.eval()


torch.cuda.empty_cache()
with torch.no_grad():
    out = model(data)
print(type(out))
print(out.size())
del out, data
torch.cuda.empty_cache()


