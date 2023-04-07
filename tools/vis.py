import numpy as np
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
from pathlib import Path

DATA_ROOT = Path('/mnt/e/Code/ink_data')

labels = ['train/1/mask/mask_1.npy']
labels = [DATA_ROOT / x for x in labels]

label = np.load(labels[0])
label[0, 0] = 1

# count time of check_array_all_zero
def check_array_all_zero(arr):
    return np.all(arr == 0)

import time
start = time.time()

print(check_array_all_zero(label))
print(time.time() - start)
# fig = plt.figure()
# matshow3d(volume, fig=fig, title="3D Volume")
# plt.show()