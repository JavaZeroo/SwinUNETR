# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data
from monai.data import load_decathlon_datalist
from utils.utils import get_transforms


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform, val_transform, test_transform = get_transforms(args)
    if args.test_mode:
        val_files = load_decathlon_datalist(
            datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_loader = data.DataLoader(
            val_ds, batch_size=8, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        loader = val_loader
    else:
        datalist = load_decathlon_datalist(
            datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.SmartCacheDataset(
                data=datalist, transform=train_transform, cache_rate=args.cache_rate, num_init_workers=args.workers
            )
        train_loader = data.DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(
            datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_loader = data.DataLoader(
            val_ds, batch_size=8, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
