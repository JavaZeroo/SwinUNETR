# import os
# import shutil
# import SimpleITK as sitk  # noqa: N813
# import numpy as np
# import itk
# import tempfile
# import monai
# from monai.data import PILReader
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
# from monai.config import print_config
from monai.networks.nets import SwinUNETR
import torch

files = [
    "train/1/surface_volume/00.tif",
    "train/1/surface_volume/01.tif",
    "train/1/surface_volume/02.tif",
    "train/1/surface_volume/03.tif",
    "train/1/surface_volume/04.tif",
    # "train/1/surface_volume/05.tif",
    # "train/1/surface_volume/06.tif",
    # "train/1/surface_volume/07.tif",
    # "train/1/surface_volume/08.tif",
    # "train/1/surface_volume/09.tif",
    # "train/1/surface_volume/10.tif",
    # "train/1/surface_volume/11.tif",
    # "train/1/surface_volume/12.tif",
    # "train/1/surface_volume/13.tif",
    # "train/1/surface_volume/14.tif",
    # "train/1/surface_volume/15.tif",
    # "train/1/surface_volume/16.tif",
    # "train/1/surface_volume/17.tif",
    # "train/1/surface_volume/18.tif",
    # "train/1/surface_volume/19.tif",
    # "train/1/surface_volume/20.tif",
    # "train/1/surface_volume/21.tif",
    # "train/1/surface_volume/22.tif",
    # "train/1/surface_volume/23.tif",
    # "train/1/surface_volume/24.tif",
    # "train/1/surface_volume/25.tif",
    # "train/1/surface_volume/26.tif",
    # "train/1/surface_volume/27.tif",
    # "train/1/surface_volume/28.tif",
    # "train/1/surface_volume/29.tif",
    # "train/1/surface_volume/30.tif",
    # "train/1/surface_volume/31.tif",
    # "train/1/surface_volume/32.tif",
    # "train/1/surface_volume/33.tif",
    # "train/1/surface_volume/34.tif",
    # "train/1/surface_volume/35.tif",
    # "train/1/surface_volume/36.tif",
    # "train/1/surface_volume/37.tif",
    # "train/1/surface_volume/38.tif",
    # "train/1/surface_volume/39.tif",
    # "train/1/surface_volume/40.tif",
    # "train/1/surface_volume/41.tif",
    # "train/1/surface_volume/42.tif",
    # "train/1/surface_volume/43.tif",
    # "train/1/surface_volume/44.tif",
    # "train/1/surface_volume/45.tif",
    # "train/1/surface_volume/46.tif",
    # "train/1/surface_volume/47.tif",
    # "train/1/surface_volume/48.tif",
    # "train/1/surface_volume/49.tif",
    # "train/1/surface_volume/50.tif",
    # "train/1/surface_volume/51.tif",
    # "train/1/surface_volume/52.tif",
    # "train/1/surface_volume/53.tif",
    # "train/1/surface_volume/54.tif",
    # "train/1/surface_volume/55.tif",
    # "train/1/surface_volume/56.tif",
    # "train/1/surface_volume/57.tif",
    # "train/1/surface_volume/58.tif",
    # "train/1/surface_volume/59.tif",
    # "train/1/surface_volume/60.tif",
    # "train/1/surface_volume/61.tif",
    # "train/1/surface_volume/62.tif",
    # "train/1/surface_volume/63.tif",
    # "train/1/surface_volume/64.tif"
]
print("Loading data...")
data, meta = LoadImage(image_only=False, reader="PILReader")(files)
print("Finished Loading")

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=False,
)
print(f"image data shape: {data.shape}")
print(f"{type(data)}")

model.eval()
with torch.no_grad():
    out = model(data)

print(type(out))
print(out.size())