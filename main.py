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

import argparse
import os
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, FocalLoss, DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.visualize import matshow3d
from utils.my_acc import FBetaScore

from utils.myModel import MyModel, MyModel2d, MyModel3dunet, MyFlexibleUNet2d, MyFlexibleUNet2dLSTM, MyBasicUNetPlusPlus, MyFlexibleUNet2dMultiScaleLSTM, MyFlexibleUNet3dMultiScaleLSTM, Net, MyBasicUNetPlusPlus2d
from utils.my_loss import CustomWeightedDiceCELoss, CustomWeightedFocalLoss

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/root/autodl-tmp/data_split", type=str, help="dataset directory")
parser.add_argument("--json_list", default="/root/autodl-tmp/data_split/data_split.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default=None,
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=65, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=0.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=65535.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=255.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=512, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=512, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=65, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
# parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")

# parser.add_argument("--focalLoss", action="store_true", help="use FocalLoss")
parser.add_argument("--num_channel", default=65, type=int, help="num of copy channels")
parser.add_argument("--num_samples", default=16, type=int, help="num of samples of transform")
parser.add_argument("--cache_rate", default=0.4, type=float, help="cache_rate")
parser.add_argument("--loss_weight", default=(2.0, 1.0), type=tuple, help="cache_rate")
parser.add_argument("--model_mode", default="3dswin", help="model_mode ['3dswin', '2dswin', '3dunet', '2dunet', '2dfunet', '2dfunetlstm', '3dunet++', 'kaggle]")
parser.add_argument("--loss_mode", default="custom", help="loss_mode ['custom', 'focalLoss', 'squared_dice', 'DiceCELoss']")
parser.add_argument("--eff", default="b5", help="efficientnet-['b0', 'b1', 'b2', 'b3', 'b4', 'b5']")
parser.add_argument("--debug", action="store_true", help="debug mode")

parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--normal", action="store_true", help="use monai Dataset class")
parser.add_argument("--mid", default=None, type=int, help="num of samples of transform")
parser.add_argument("--threshold", default=0.4, type=int, help="num of samples of transform")
parser.add_argument("--z_range_0", default=None, type=int, help="num of samples of transform")
parser.add_argument("--z_range_1", default=None, type=int, help="num of samples of transform")
parser.add_argument("--add_shuffled", default=0, type=int, help="num of add_shuffled")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.use_normal_dataset = args.normal if args.normal else args.use_normal_dataset
    if args.z_range_0 is not None and args.z_range_1 is not None:
        args.roi_z = args.z_range_1 - args.z_range_0
        args.z_range = [args.z_range_0, args.z_range_1]
    else:
        args.z_range = [args.mid - args.roi_z // 2, args.mid + args.roi_z // 2]
    args.logdir = f"{args.roi_x}_{args.model_mode}_{args.eff}_{args.roi_z}_mid{args.mid}_{args.optim_name}_{time.strftime('%b-%dd_%Hh-%Mm', time.localtime(time.time()))}"
    args.logdir = "./runs/" + args.logdir if not args.debug else './debug'
    args.num_channel = args.roi_z
    if args.debug:
        args.val_every = 1
        args.use_normal_dataset = False
        args.cache_rate = 0.1
    main_worker(gpu=0, args=args)


def main_worker(gpu, args):

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    
    # 定义模型
    if args.model_mode == "3dswin":
        model = MyModel(img_size=(args.roi_x,args.roi_y,args.roi_y))
    elif args.model_mode == "2dswin":
        model = MyModel2d(img_size=(args.roi_x,args.roi_y))
    elif args.model_mode == "3dunet":
        model = MyModel3dunet()
    elif args.model_mode == "2dfunet":
        model = MyFlexibleUNet2d(args)
    elif args.model_mode == "2dfunetlstm":
        model = MyFlexibleUNet2dMultiScaleLSTM(args)
    elif args.model_mode == "3dfunetlstm":
        model = MyFlexibleUNet3dMultiScaleLSTM(args)
    elif args.model_mode == "3dunet++":
        model = MyBasicUNetPlusPlus(args)
    elif args.model_mode == "2dunet++":
        model = MyBasicUNetPlusPlus2d(args)
    elif args.model_mode == "kaggle":
        model = Net(args)
    else:
        raise ValueError("model mode error")

    # 如果要接着训练的话，添加模型的名字。然后模型放到pretrained文件夹
    if args.pretrained_model_name is not None:
            # raise ValueError("2d model can not resume from ckpt")
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        if args.model_mode in ["2dswin", "3dunet", "2dfunet", "2dfunetlstm", "3dunet++", "3dfunetlstm", "kaggle", "2dunet++"]:
            model.load_state_dict(model_dict)
        elif args.model_mode == "3dswin":
            model.load_swin_ckpt(model_dict)
        else:
            raise ValueError("model mode error")
        print("Use pretrained weights")

    # 选择loss函数
    if args.loss_mode == 'focalLoss':
        loss = CustomWeightedFocalLoss(ink_weight=3.0, weight=args.loss_weight)
        # loss = FocalLoss(weight=[10.0])
    elif args.loss_mode == 'squared_dice':
        loss = DiceCELoss(squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
    elif args.loss_mode == 'DiceCELoss':
        loss = DiceCELoss(include_background=True, sigmoid=True, ce_weight=torch.Tensor([ 10])).cuda(0) # Normally
    elif args.loss_mode == 'custom':
        loss = CustomWeightedDiceCELoss(ink_weight=1.5, weight=args.loss_weight)
    elif args.loss_mode == 'DiceFocalLoss':
        loss = DiceFocalLoss(weight=args.loss_weight)
    elif args.loss_mode == 'bce':
        loss = torch.nn.BCEWithLogitsLoss()
        
    # acc函数
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    miou_acc = MeanIoU(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    f05_acc = FBetaScore(beta=0.5, include_background=True)

    # 模型推理，主要是validation的时候用的
    if args.model_mode in ["3dswin", "3dunet", "3dunet++", "3dfunetlstm"]:
        model_inferer = partial(
            sliding_window_inference,
            roi_size = (args.roi_x,args.roi_y,args.roi_z),
            sw_batch_size = 8,
            predictor = model,
            overlap = 0.5,
            progress = True,
            padding_mode = "reflect", 
            device = "cpu", 
            sw_device = "cuda"
        )
    elif args.model_mode in ["2dswin", "2dfunet", "2dfunetlstm", "kaggle", "2dunet++"]:
        model_inferer = partial(
            sliding_window_inference,
            roi_size = (args.roi_x,args.roi_y),
            sw_batch_size = 8,
            predictor = model,
            overlap = 0.5,
            progress = True,
            padding_mode = "reflect", 
            device = "cpu", 
            sw_device = "cuda"
        )     
    else:
        raise ValueError("model mode error")
    
    # 不重要 
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(0)
    ########################################################################
    
    # 优化器 选择
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    # 学习率调整策略
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
        
        
    print("Lodaer test")
    for i in loader[0]:
        print(i['image'].shape)
        print(i['label'].shape)
        print(i['inklabels'].shape)
        print(torch.unique(i['image']))
        print(torch.unique(i['label']))
        print(torch.unique(i['inklabels']))
        break
    print("Pass Test")
    print(args)
    
    # 训练
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss,
        acc_func=f05_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
