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

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
from utils.utils import dice, resample_3d, resample_2d, binary

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.data import decollate_batch
from utils.my_acc import FBetaScore

from utils.myModel import MyModel,MyModel2d, MyFlexibleUNet2dLSTM

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
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
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
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=0.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=65535.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=256.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=256, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=256, type=int, help="roi size in y direction")
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
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")

parser.add_argument("--focalLoss", action="store_true", help="use FocalLoss")
parser.add_argument("--num_channel", default=65, type=int, help="num of copy channels")
parser.add_argument("--num_samples", default=16, type=int, help="num of samples of transform")
parser.add_argument("--cache_rate", default=0.4, type=float, help="cache_rate")
parser.add_argument("--loss_weight", default=(2.0, 1.0), type=tuple, help="cache_rate")
parser.add_argument("--loss_mode", default="custom", help="loss_mode ['custom', 'focalLoss', 'squared_dice', 'DiceCELoss']")
parser.add_argument("--eff", default="b5", help="efficientnet-['b0', 'b1', 'b2', 'b3', 'b4', 'b5']")
parser.add_argument("--debug", action="store_true", help="debug mode")

parser.add_argument("--exp_name", default="test2", type=str, help="experiment name")
parser.add_argument("--model_mode", default="3dswin", help="model_mode ['3dswin', '2dswin', '3dunet', '2dunet']")
parser.add_argument("--normal", action="store_true", help="use monai Dataset class")
parser.add_argument("--mid", default=None, type=int, help="num of samples of transform")
parser.add_argument("--threshold", default=0.5, type=int, help="num of samples of transform")

def main():
    args = parser.parse_args()
    args.test_mode = True
    args.num_channel = args.roi_z
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.model_mode == "3dswin":
        model = MyModel(img_size=(args.roi_x,args.roi_y,args.roi_y))
    elif args.model_mode == "2dswin":
        model = MyModel2d(img_size=(args.roi_x,args.roi_y))
    elif args.model_mode == "2dfunetlstm":
        model = MyFlexibleUNet2dLSTM(args)
    else:
        raise ValueError("model mode error")
    f05_acc = FBetaScore(beta=0.5, include_background=True)

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    print(args)
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"][ :, 0:1, :, :].cuda())

            _, d, h, w = val_labels.shape
            target_shape = (h, w)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, 
                (args.roi_x, args.roi_y), 
                4, 
                model, 
                overlap = 0,
                progress = True,
                padding_mode = "reflect", 
                device = "cpu", 
                sw_device = "cuda"
            )
            print(val_outputs.shape)
            if not val_outputs.is_cuda:
                val_labels = val_labels.cpu()
            val_outputs_bin = binary(val_outputs, threshold=args.threshold)
            val_labels_bin = binary(val_labels, threshold=args.threshold)

            np.save("binary.npy", val_outputs_bin)
            val_outputs_list = decollate_batch(val_outputs_bin)
            # val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_labels_list = decollate_batch(val_labels_bin)
            # val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            f05_acc.reset()
            f05_acc(y_pred=val_outputs_list, y=val_labels_list)
            acc = f05_acc.aggregate()[0]


            # val_outputs = torch.softmax(val_outputs, 1).cpu()
            val_outputs = np.array(val_outputs)[0]
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu()
            val_labels = np.array(val_labels)[0, 0, :, :]
            if args.model_mode in ["2dswin", "2dfunetlstm"]:
                val_outputs = resample_2d(val_outputs, target_shape)
            elif args.model_mode == "3dswin":
                val_outputs = resample_3d(val_outputs, target_shape)
            else:
                raise ValueError("model_mode should be ['3dswin', '2dswin', '3dunet', '2dunet', '2dfunetlstm']")
            
            dice_list_sub = []
            for i in [1]:
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("acc: {}".format(acc))
            dice_list_case.append(mean_dice)
            np.save(
                os.path.join(output_directory, img_name), val_outputs[:,:]
            )

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))


if __name__ == "__main__":
    main()
