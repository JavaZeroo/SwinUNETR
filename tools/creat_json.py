import json
from pathlib import Path
from tqdm import tqdm
OUTPUT = Path(r'my.json')
DATA_ROOT = Path(r'/mnt/e/Code/vesuvius-challenge-ink-detection')
trains = DATA_ROOT / 'train'
tests = DATA_ROOT / 'test'

# list of all subfolder of trains
trains_folders = [f for f in trains.iterdir() if f.is_dir()]
# list of all subfolder of tests
tests_folders = [f for f in tests.iterdir() if f.is_dir()]

js = dict({'training': [], 'validation': [],'testing': [], 
        "labels": {
        "0": "background",
        "1": "ink"
    }})

for sub_train in tqdm(trains_folders):
    # list of all files in subfolder
    sub_train_files = [str(f) for f in (sub_train / 'surface_volume').iterdir() if f.is_file()]
    js['training'].append({'image': sub_train_files, 'label': str(sub_train / 'mask.png'), 'inklabels': str(sub_train / 'inklabels.png')})
    
for sub_test in tqdm(tests_folders):
    # list of all files in subfolder
    sub_test_files = [str(f) for f in (sub_test / 'surface_volume').iterdir() if f.is_file()]
    js['testing'].append({'image': sub_test_files,  'label': str(sub_train / 'mask.png')})
    
with OUTPUT.open('w') as f:
    json.dump(js, f)