import os
import shutil
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
import tempfile
import monai
from monai.data import PILReader
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
from monai.config import print_config


files = [
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/00.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/01.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/02.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/03.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/04.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/05.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/06.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/07.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/08.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/09.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/10.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/11.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/12.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/13.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/14.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/15.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/16.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/17.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/18.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/19.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/20.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/21.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/22.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/23.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/24.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/25.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/26.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/27.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/28.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/29.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/30.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/31.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/32.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/33.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/34.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/35.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/36.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/37.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/38.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/39.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/40.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/41.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/42.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/43.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/44.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/45.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/46.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/47.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/48.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/49.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/50.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/51.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/52.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/53.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/54.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/55.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/56.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/57.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/58.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/59.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/60.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/61.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/62.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/63.tif",
    "/mnt/e/Code/vesuvius-challenge-ink-detection/train/1/surface_volume/64.tif"
]

data, meta = LoadImage(image_only=False, reader="PILReader")(files)

print(f"image data shape: {data.shape}")
print(f"{type(data)}")