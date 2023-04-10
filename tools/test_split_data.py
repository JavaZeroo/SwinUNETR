from pathlib import Path
import numpy as np
import torch
from monai import transforms


DATA_ROOT = Path("/root/autodl-tmp/data_split")

# for i in DATA_ROOT.iterdir():
#     print(i)
for i in range(1,10):
    test_mask = DATA_ROOT / f'mask_{i}.npy'
    mask = transforms.LoadImage(image_only=True)(test_mask)
    tensor_mask = transforms.ToTensor()(mask)
    print(torch.unique(tensor_mask))
    
for i in range(1,10):
    test_mask = DATA_ROOT / f'inklabels_{i}.npy'
    mask = transforms.LoadImage(image_only=True)(test_mask)
    tensor_mask = transforms.ToTensor()(mask)
    print(torch.unique(tensor_mask))