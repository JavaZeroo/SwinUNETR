# %%
from pathlib import Path
import numpy as np

# %%
DATA_ROOT = Path("/root/autodl-fs/ink_data/")
OUTPUT = Path("/root/autodl-tmp/data_split")

# %%
train_folder = DATA_ROOT / "train"
test_folder = DATA_ROOT / "test"
train_1 = train_folder / "1"
train_2 = train_folder / "2"
train_3 = train_folder / "3"


# %%
# train_1_img = train_1 / "surface_volume.npy"
# train_1_inklabel = train_1 / "inklabels.npy"
# train_1_ir = train_1 / "ir.npy"
# train_1_mask = train_1 / "mask.npy"

# train_2_img = train_2 / "surface_volume.npy"
# train_2_inlabel = train_2 / "inklabels.npy"
# train_2_ir = train_2 / "ir.npy"
# train_2_mask = train_2 / "mask.npy"

# train_3_img = train_3 / "surface_volume.npy"
# train_3_inlabel = train_3 / "inklabels.npy"
# train_3_ir = train_3 / "ir.npy"
# train_3_mask = train_3 / "mask.npy"

# test_1_img = test_folder / "1" / "surface_volume.npy"

count = 0
OUTPUT.mkdir(exist_ok=True)
folders = list(train_folder.iterdir())
from tqdm import tqdm
for i in tqdm(folders):
    # img, inklabel, ir, mask need to be split from (a, b) to (a/2, b)
    if i.name != '2':
        for file in i.iterdir():
            read_file = np.load(file)
            if len(read_file.shape) == 2:
                x, y = read_file.shape
                print(x, y)
                read_file_1 = read_file[:int(x/3), :]
                read_file_2 = read_file[int(x/3):int(2 * x / 3), :]
                read_file_3 = read_file[int(2*x/3):,:]
                np.save(OUTPUT / f"{file.stem}_{count+1}.npy", read_file_1)
                np.save(OUTPUT / f"{file.stem}_{count+2}.npy", read_file_2)
                np.save(OUTPUT / f"{file.stem}_{count+3}.npy", read_file_3)
            else:
                h, x, y = read_file.shape
                print(x, y)
                read_file_1 = read_file[:, :int(x/3), :]
                read_file_2 = read_file[:, int(x/3):int(2*x/3), :]
                read_file_3 = read_file[:, int(2*x/3):, :]
                np.save(OUTPUT / f"{file.stem}_{count+1}.npy", read_file_1)
                np.save(OUTPUT / f"{file.stem}_{count+2}.npy", read_file_2)
                np.save(OUTPUT / f"{file.stem}_{count+3}.npy", read_file_3)
        count += 3
        continue
    elif i.name == '2':
        for file in i.iterdir():
            read_file = np.load(file)
            if len(read_file.shape) == 2:
                x, y = read_file.shape
                print(x, y)
                read_file_1 = read_file[:int(x/4), :]
                read_file_2 = read_file[int(x/4):int(x/2), :]
                read_file_3 = read_file[int(x/2):int(3*x/4),:]
                read_file_4 = read_file[int(3*x/4):,:]
                np.save(OUTPUT / f"{file.stem}_{count+1}.npy", read_file_1[:, :7000])
                np.save(OUTPUT / f"{file.stem}_{count+2}.npy", read_file_2[:, :8000])
                np.save(OUTPUT / f"{file.stem}_{count+3}.npy", read_file_3)
                np.save(OUTPUT / f"{file.stem}_{count+4}.npy", read_file_4)
            else:
                h, x, y = read_file.shape
                print(x, y)
                read_file_1 = read_file[:,:int(x/4), :]
                read_file_2 = read_file[:,int(x/4):int(x/2), :]
                read_file_3 = read_file[:,int(x/2):int(3*x/4),:]
                read_file_4 = read_file[:,int(3*x/4):,:]
                np.save(OUTPUT / f"{file.stem}_{count+1}.npy", read_file_1[:, :, :7000])
                np.save(OUTPUT / f"{file.stem}_{count+2}.npy", read_file_2[:, :, :8000])
                np.save(OUTPUT / f"{file.stem}_{count+3}.npy", read_file_3)
                np.save(OUTPUT / f"{file.stem}_{count+4}.npy", read_file_4)
        count += 4
        continue
    
    
    


