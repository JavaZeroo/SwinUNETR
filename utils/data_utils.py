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

from monai import data, transforms
from monai.data import load_decathlon_datalist

from utils.my_transform import *


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    if not args.model2d:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label", 'inklabels'], reader="NumpyReader"),
                transforms.AddChanneld(keys=["image"]),
                Copyd(keys=["label", 'inklabels'], num_channel=args.num_channel, add_channel=True), 
                change_channeld(keys=["image", "label", 'inklabels']),
                transforms.Orientationd(keys=["image", "label", 'inklabels'], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label", 'inklabels'], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                Drop1Layerd(keys=["image", "label", 'inklabels']),
                printShaped(keys=["image", "label", 'inklabels']),
                transforms.CropForegroundd(keys=["image", "label", 'inklabels'], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label", 'inklabels'],
                    label_key="inklabels",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=8,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=False,
                ),

                transforms.RandFlipd(keys=["image", 'inklabels'], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", 'inklabels'], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", 'inklabels'], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", 'inklabels'], prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", 'inklabels']),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label", 'inklabels'], reader="NumpyReader"),
                Copyd(keys=["label", 'inklabels'], num_channel=65), 
                # transforms.GridSplitd(keys=["image", 'inklabels'], grid=(10,10)),
                transforms.AddChanneld(keys=["image", "label", 'inklabels']),
                transforms.Orientationd(keys=["image", "label", 'inklabels'], axcodes="RAS"),
                change_channeld(keys=["image", "label", 'inklabels']),
                Drop1Layerd(keys=["image", "label", 'inklabels']),
                transforms.Spacingd(
                    keys=["image", "label", 'inklabels'], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label", 'inklabels'], source_key="image"),
                transforms.ToTensord(keys=["image", 'inklabels']),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], reader="NumpyReader"),
                transforms.AddChanneld(keys=["image"]),
                Copyd(keys=["label", 'inklabels'], num_channel=args.num_channel), 
                # transforms.Orientationd(keys=["image"], axcodes="RAS"),
                change_channeld(keys=["image", "label", 'inklabels']),
                transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ToTensord(keys=["image"]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label", 'inklabels'], reader="NumpyReader"),
                transforms.AddChanneld(keys=["image"]),
                Copyd(keys=["label", 'inklabels'], num_channel=args.num_channel, add_channel=True), 
                change_channeld(keys=["image", "label", 'inklabels']),
                transforms.Orientationd(keys=["image", "label", 'inklabels'], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label", 'inklabels'], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                # Drop1Layerd(keys=["image", "label", 'inklabels']),
                transforms.CropForegroundd(keys=["image", "label", 'inklabels'], source_key="image"),
                # printShaped(keys=["image", "label", 'inklabels']),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label", 'inklabels'],
                    label_key="inklabels",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=False,
                ),
                # printShaped(keys=["image", "label", 'inklabels']),
                transforms.RandFlipd(keys=["image", 'inklabels'], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", 'inklabels'], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", 'inklabels'], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", 'inklabels'], prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                change_channeld(keys=["image", "label", 'inklabels'], back=True),
                # printShaped(keys=["image", "label", 'inklabels']),
                remove_channeld(keys=["image", "label", 'inklabels']),
                transforms.ToTensord(keys=["image", 'inklabels']),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label", 'inklabels'], reader="NumpyReader"),
                Copyd(keys=["label", 'inklabels'], num_channel=args.num_channel), 
                # transforms.GridSplitd(keys=["image", 'inklabels'], grid=(10,10)),
                transforms.AddChanneld(keys=["image", "label", 'inklabels']),
                transforms.Orientationd(keys=["image", "label", 'inklabels'], axcodes="RAS"),
                change_channeld(keys=["image", "label", 'inklabels']),
                # Drop1Layerd(keys=["image", "label", 'inklabels']),
                transforms.Spacingd(
                    keys=["image", "label", 'inklabels'], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label", 'inklabels'], source_key="image"),
                change_channeld(keys=["image", "label", 'inklabels'], back=True),
                remove_channeld(keys=["image", "label", 'inklabels']),
                transforms.ToTensord(keys=["image", 'inklabels']),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], reader="NumpyReader"),
                # transforms.AddChanneld(keys=["image"]),
                Copyd(keys=["label", 'inklabels'], num_channel=args.num_channel), 
                # transforms.Orientationd(keys=["image"], axcodes="RAS"),
                # change_channeld(keys=["image", "label", 'inklabels']),
                transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        

    if args.test_mode:
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=8, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = val_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.SmartCacheDataset(
                data=datalist, transform=train_transform, cache_rate=0.5, num_init_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=8, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
