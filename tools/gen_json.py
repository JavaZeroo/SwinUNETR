import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

DATA_ROOT = Path(r'/root/autodl-tmp/MyData')

js = {
    "training": [],
    "validation": [],
    "testing": []
}
all_train = []
for i in tqdm(range(90)):
    for j in range(1, 4):
        tmp_js = {
            "image":[f"train/{j}/surface_volume/{str(i)}.npy"],
            "label":[f"train/{j}/mask/{str(i)}.npy"], 
            "ir":[f"train/{j}/ir/{str(i)}.npy"],
            "inklabels":[f"train/{j}/inklabels/{str(i)}.npy"]
        }
        mask = np.load(DATA_ROOT / tmp_js["label"][0])
        # check if mask is all 0
        if np.all(mask == 0):
            continue
        # check if mask has 30% numbers is 0
        elif np.sum(mask == 0) / mask.size > 0.3:
            continue
        
        all_train.append(tmp_js)
print(len(all_train))
# random choose 30% for validation
import random
random.shuffle(all_train)
val_len = int(len(all_train) * 0.3)
js["training"] = all_train[val_len:]
js["validation"] = all_train[:val_len]

for i in range(10):
    tmp_js = {
        "image":[f"test/{i+1}/surface_volume/{str(i)}.npy"],
        "label":[f"test/{i+1}/mask/{str(i)}.npy"]
    }
    js["testing"].append(tmp_js)

with open(DATA_ROOT / "train.json", "w") as f:
    json.dump(js, f)