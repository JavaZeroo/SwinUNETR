# %%
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from tifffile import tifffile

import numpy as np

# # %%
# Split image in to (x, y)

def split(img):
    imgs = []    
    for i in range(30):
        for j in range(27):
            imgs.append(img[i*211:(i+1)*211, j*303:(j+1)*303])
    return imgs

# %%
x = 211 # 6330/211=30
y = 303 # 8181/303=27
DATA_ROOT = Path(r'/root/autodl-tmp/vesuvius-challenge-ink-detection/')
OUTPUT = Path(r'/root/autodl-fs/MyData')

# DATA_ROOT=Path('/mnt/e/Code/vesuvius-challenge-ink-detection')
# OUTPUT = Path('/mnt/e/Code/ink_data')

trains = DATA_ROOT / 'train'
tests = DATA_ROOT / 'test'

# list of all subfolder of trains
trains_folders = [f for f in trains.iterdir() if f.is_dir()]
# list of all subfolder of tests
tests_folders = [f for f in tests.iterdir() if f.is_dir()]

# js = dict({'training': [], 'validation': [],'testing': [], 
#         "labels": {
#         "0": "background",
#         "1": "ink"
#     }})
sub_train_files = []
train_output = OUTPUT / "train"
test_output = OUTPUT / "test"
for i, sub_train in enumerate(trains_folders):
    print(sub_train)
    # list of all files in subfolder
    img_root = train_output / str(i+1) / "imgs"
    mask_root = train_output / str(i+1) / "mask"
    ink_label_root = train_output / str(i+1) / "ink_label"
    img_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(exist_ok=True)
    ink_label_root.mkdir(exist_ok=True)
    filenames = [ f for f in (sub_train / 'surface_volume').iterdir()]
    for f in tqdm(filenames):
        image_sub_folder = img_root / str(f.stem)
        image_sub_folder.mkdir(exist_ok=True)
        img = tifffile.imread(f)
        imgs = split(img)
        del img
        for j, image in enumerate(imgs):
            np.save(str(image_sub_folder / (f.stem +f"_{j+1}.npy")), image)
            
    with Image.open(str(sub_train / 'mask.png')) as m:
        mask_img = np.array(m)
    masks = split(mask_img)
    del mask_img
    for j, image in enumerate(masks):
        np.save(str(mask_root / (f"mask_{j+1}.npy")), image)
        del image
        
    with Image.open(str(sub_train / 'inklabels.png')) as ink:
        ink_img = np.array(m)
    inks = split(ink_img)
    del ink_img
    for j, image in enumerate(inks):
        np.save(str(ink_label_root / (f"ink_label_{j+1}.npy")), image)
        del image

    # js['training'].append({'image': sub_train_files, 'label': str(sub_train / 'mask.png'), 'inklabels': str(sub_train / 'inklabels.png')})
sub_test_files = []
for i, sub_test in enumerate(tests_folders):
    print(sub_test)
    # list of all files in subfolder
    img_root = test_output / str(i+1) / "imgs"
    mask_root = test_output / str(i+1) / "mask"
    ink_label_root = test_output / str(i+1) / "ink_label"
    img_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(exist_ok=True)
    ink_label_root.mkdir(exist_ok=True)
    filenames = [ f for f in (sub_test / 'surface_volume').iterdir()]
    for f in tqdm(filenames):
        image_sub_folder = img_root / str(f.stem)
        image_sub_folder.mkdir(exist_ok=True)
        img = tifffile.imread(f)
        imgs = split(img)
        del img
        for j, image in enumerate(imgs):
            np.save(str(image_sub_folder / (f.stem +f"_{j+1}.npy")), image)
        del imgs
        
    with Image.open(str(sub_test / 'mask.png')) as m:
        mask_img = np.array(m)
    masks = split(mask_img)
    del mask_img
    for j, image in enumerate(masks):
        np.save(str(mask_root / (f"mask_{j+1}.npy")), image)
        del image
    # js['testing'].append({'image': sub_test_files,  'label': str(sub_test / 'mask.png')})

# %%
def getfolder(index):
    if index >= 10:
        return str(index)
    else:
        return f"0{index}"
import json
# Generate train json
js = dict({'training': [], 'validation': [],'testing': [], 
        "labels": {
        "0": "background",
        "1": "ink"
    }})
# train 0:162 163:324
OUTPUT_train = OUTPUT / "train"
OUTPUT_test = OUTPUT / "test"
for data in OUTPUT_train.iterdir():
    export_root = Path("train") / data.name
    images_folder = export_root / "imgs"
    mask_folder = export_root / "mask"
    ink_label_folder = export_root / "ink_label"
    
    for i in range(1,811):
        mask = np.load(data / "mask" / f'mask_{i}.npy')
        if np.all(mask == 0):
            continue
        if 162 < i <= 324:
            sub_train_files = []
            for j in range(65):
                # val 
                sub_train_files.append(str(images_folder / getfolder(j) / f"{getfolder(j)}_{i}.npy"))
            js['validation'].append({'image': sub_train_files, 'label': str(mask_folder / f'mask_{i}.npy'), 'inklabels': str(ink_label_folder / f'ink_label_{i}.npy')})
        else:
            # train
            sub_train_files = []
            for j in range(65):
                # val 
                sub_train_files.append(str(images_folder / getfolder(j) / f"{getfolder(j)}_{i}.npy"))
            js['training'].append({'image': sub_train_files, 'label': str(mask_folder / f'mask_{i}.npy'), 'inklabels': str(ink_label_folder / f'ink_label_{i}.npy')})
# test
for data in OUTPUT_test.iterdir():
    export_root = Path("test") / data.name
    images_folder = export_root / "imgs"
    mask_folder = export_root / "mask"
    ink_label_folder = export_root / "ink_label"
    for i in range(1,811):
        if np.all(mask == 0):
            continue
        sub_test_files = []
        for j in range(65):
            # val 
            sub_test_files.append(str(images_folder / getfolder(j) / f"{getfolder(j)}_{i}.npy"))
        js['testing'].append({'image': sub_test_files, 'label': str(mask_folder / f'mask_{i}.npy')})   
json_file = OUTPUT / "my.json"
with open(json_file, 'w') as f:
    json.dump(js, f)




