from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from tifffile import tifffile

DATA_ROOT=Path('/mnt/e/Code/vesuvius-challenge-ink-detection')
OUTPUT = Path('/mnt/e/Code/ink_data')

TRAIN_ROOT = DATA_ROOT / 'train'
TEST_ROOT = DATA_ROOT / 'test'

# create tqdm progress bar
pbar = tqdm(total=len(list(TRAIN_ROOT.iterdir())))

def getfolder(index):
    if index >= 10:
        return str(index)
    else:
        return f"0{index}"

for i in TRAIN_ROOT.iterdir():
    out = OUTPUT / 'train' / i.name
    # mkdir for out
    out.mkdir(parents=True, exist_ok=True)
    
    surface_volume_folder = i / 'surface_volume'
    mask = i / 'mask.png'
    ir = i / 'ir.png'
    inklabels = i / 'inklabels.png'
    # read all image in surface_volume folder and save as numpy array
    if i.name != '1':
        surface_volume = np.array([np.array(tifffile.imread(surface_volume_folder / f'{getfolder(x)}.tif')) for x in range(65)])
        np.save(out / 'surface_volume.npy', surface_volume)
    # read mask, ir and inklabels and save as numpy array
    np.save(out / mask.with_suffix('.npy'), np.array(Image.open(mask)))
    np.save(out / ir.with_suffix('.npy'), np.array(Image.open(ir)))
    np.save(out / inklabels.with_suffix('.npy'), np.array(Image.open(inklabels)))
    pbar.update(1)
    
    
# do all the same for test data
pbar = tqdm(total=len(list(TEST_ROOT.iterdir())))

for i in TEST_ROOT.iterdir():
    out = OUTPUT / 'test' / i.name
    # mkdir for out
    out.mkdir(parents=True, exist_ok=True)
    
    surface_volume_folder = i / 'surface_volume'
    mask = i / 'mask.png'
    ir = i / 'ir.png'
    # read all image in surface_volume folder and save as numpy array
    surface_volume = np.array([np.array(tifffile.imread(surface_volume_folder / f'{getfolder(x)}.tif')) for x in range(65)])
    np.save(out / 'surface_volume.npy', surface_volume)
    # read mask, ir and inklabels and save as numpy array
    np.save(out / mask.with_suffix('.npy'), np.array(Image.open(mask)))
    np.save(out / ir.with_suffix('.npy'), np.array(Image.open(ir)))
    pbar.update(1)