from tifffile import tifffile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

DATA_ROOT = Path("/mnt/e/Code/vesuvius-challenge-ink-detection")
DATA_ROOT = Path("e:/Code/vesuvius-challenge-ink-detection")
TRAIN_ROOT = DATA_ROOT / "train"

train_folders = [f for f in TRAIN_ROOT.iterdir() if f.is_dir()]

def get_tif(folder, index):
    if index < 10:
        index = f"0{index}"
    else:
        index = str(index)
    return np.array(tifffile.imread(folder / 'surface_volume' / f"{index}.tif"))

def get_ink(folder):
    return np.array(Image.open(folder / "inklabels.png"))


maximum = 2**10
bit = 64

for train_folder in train_folders:
    for i in tqdm(range(65)):
        img_data = get_tif(train_folder, i)
        
        img_data = (img_data / bit)
        if maximum <= 2**8:
            img_data = img_data.astype(np.uint8)
        histogram, _ = np.histogram(img_data * -(1-get_ink(train_folder)), bins=maximum, range=(0, maximum))
        histogram[0] = 0
        
        histogram_masked, _ = np.histogram(img_data * get_ink(train_folder), bins=maximum, range=(0, maximum))
        histogram_masked[0] = 0
        
        del img_data
        
        total_pixels = np.sum(histogram)
        frequency = histogram / total_pixels
        
        total_pixels = np.sum(histogram_masked)
        frequency_masked = histogram_masked / total_pixels
        
        max_frequency = np.max(frequency)
        max_pixel_value = np.argmax(frequency)
        
        max_frequency_masked = np.max(frequency_masked)
        max_pixel_value_masked = np.argmax(frequency_masked)
        
        plt.figure()
        plt.bar(range(maximum), frequency, color='red', alpha=0.5)
        plt.axvline(max_pixel_value, color='red', linestyle='dashed', linewidth=1)
        plt.text(max_pixel_value+5, max_frequency, f'num: {max_pixel_value}\nfeq: {max_frequency:.4f}', fontsize=10)

        plt.bar(range(maximum), frequency_masked, color='blue', alpha=0.5)
        plt.axvline(max_pixel_value_masked, color='blue', linestyle='dashed', linewidth=1)
        plt.text(max_pixel_value_masked+5, max_frequency_masked, f'num: {max_pixel_value_masked}\nfeq: {max_frequency_masked:.4f}', fontsize=10)
        plt.savefig(f"./anaysis/{maximum}_{i}.png")
        # break
    break