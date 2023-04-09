# %%
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from tifffile import tifffile

import numpy as np

# # %%
# Split image in to (x, y)

def split(img, save_dir):
    imgs = []    
    for i in range(10):
        for j in range(9):
            if len(img.shape) == 3:
                imgs.append(img[:,i*633:(i+1)*633, j*909:(j+1)*909])
            elif len(img.shape) == 2:
                imgs.append(img[i*633:(i+1)*633, j*909:(j+1)*909])
                
    for index, image in enumerate(imgs):
        np.save(save_dir / f"{index}.npy", image)
    return None

x = 633 # 6330/633=10
y = 909 # 8181/909=9
DATA_ROOT = Path(r'/root/autodl-fs/ink_data/')
OUTPUT = Path(r'/root/autodl-tmp/MyData')


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
train_output = OUTPUT / "train"
test_output = OUTPUT / "test"

sub_train_files = []
for i, sub_train in enumerate(tqdm(trains_folders)):
    print(sub_train)
    # list of all files in subfolder
    ouput_sub = train_output / str(i+1)
    img_root = train_output / str(i+1) / "surface_volume"
    mask_root = train_output / str(i+1) / "mask"
    ink_label_root = train_output / str(i+1) / "inklabels"
    ir_root = train_output / str(i+1) / "ir"
    
    img_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(exist_ok=True)
    ink_label_root.mkdir(exist_ok=True)
    ir_root.mkdir(exist_ok=True)
    
    print(type(np.load(sub_train / "mask.npy")))
    
    split(np.load(sub_train / "surface_volume.npy"), img_root)
    split(np.load(sub_train / "mask.npy"), mask_root)
    split(np.load(sub_train / "inklabels.npy"), ink_label_root)
    split(np.load(sub_train / "ir.npy"), ir_root)
    
for i, sub_test in enumerate(tqdm(tests_folders)):
    print(sub_test)
    # list of all files in subfolder
    ouput_sub = test_output / str(i+1)
    img_root = test_output / str(i+1) / "surface_volume"
    mask_root = test_output / str(i+1) / "mask"

    img_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(exist_ok=True)

    split(np.load(sub_test / "surface_volume.npy"), img_root)
    split(np.load(sub_test / "mask.npy"), mask_root)

    


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
# OUTPUT_train = OUTPUT / "train"
# OUTPUT_test = OUTPUT / "test"
# for data in OUTPUT_train.iterdir():
#     export_root = Path("train") / data.name
#     images_folder = export_root / "imgs"
#     mask_folder = export_root / "mask"
#     ink_label_folder = export_root / "ink_label"
    
#     for i in range(1,811):
#         mask = np.load(data / "mask" / f'mask_{i}.npy')
#         if np.all(mask == 0):
#             continue
#         if 162 < i <= 324:
#             sub_train_files = []
#             for j in range(65):
#                 # val 
#                 sub_train_files.append(str(images_folder / getfolder(j) / f"{getfolder(j)}_{i}.npy"))
#             js['validation'].append({'image': sub_train_files, 'label': str(mask_folder / f'mask_{i}.npy'), 'inklabels': str(ink_label_folder / f'ink_label_{i}.npy')})
#         else:
#             # train
#             sub_train_files = []
#             for j in range(65):
#                 # val 
#                 sub_train_files.append(str(images_folder / getfolder(j) / f"{getfolder(j)}_{i}.npy"))
#             js['training'].append({'image': sub_train_files, 'label': str(mask_folder / f'mask_{i}.npy'), 'inklabels': str(ink_label_folder / f'ink_label_{i}.npy')})
# # test
# for data in OUTPUT_test.iterdir():
#     export_root = Path("test") / data.name
#     images_folder = export_root / "imgs"
#     mask_folder = export_root / "mask"
#     ink_label_folder = export_root / "ink_label"
#     for i in range(1,811):
#         if np.all(mask == 0):
#             continue
#         sub_test_files = []
#         for j in range(65):
#             # val 
#             sub_test_files.append(str(images_folder / getfolder(j) / f"{getfolder(j)}_{i}.npy"))
#         js['testing'].append({'image': sub_test_files, 'label': str(mask_folder / f'mask_{i}.npy')})   
# json_file = OUTPUT / "my.json"
# with open(json_file, 'w') as f:
#     json.dump(js, f)




