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

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather, add_fig, binary


from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["inklabels"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            
            # logits是模型输出。target在加载数据的时候都被复制成了和logits一样的形状。最后输出的时候要把他弄回只有一层的
            logits = model(data)
            
            # 2d 和 3d 的张量意义不一样
            # 2d 的是(batch, channel, h, w)
            # 3d 的是(batch, channel=1, h, w, d)
            # '2d 的 channel' 和 '3d的d' 是一个东西
            if args.model_mode in ["2dswin", "2dfunet", "2dfunetlstm","2dunet++", "kaggle"]:
                logits, target = logits.cuda(0), target.cuda(0)
                loss = loss_func(logits, target[ :, 0:1, :, :])
            elif args.model_mode in ["3dswin", "3dunet++", "3dfunetlstm"]:
                if args.debug:
                    print(logits)
                loss = loss_func(logits[:, :, :, :, 0], target[:, :, :, :, 0])
            elif args.model_mode == "3dunet":
                if args.debug:
                    print(logits.shape, target[:, :, :, :, 0:1].shape)
                    print(logits.device, target[:, :, :, :, 0:1].device)
                loss = loss_func(logits, target[:, :, :, :, 0:1])
            else:
                raise ValueError("model_mode should be ['3dswin', '2dswin', '3dunet', '2dunet', 2dfunetlstm]")
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, loss_func, args, model_inferer=None, post_label=None, post_pred=None, writer=None):
    # 和 train差不多 自己对着看吧
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["inklabels"]
            if args.model_mode in ["3dswin", "3dunet", "3dunet++", "3dfunetlstm"]:
                data, target = data, target[:, :, :, :, 0:1]
            elif args.model_mode in ["2dswin", "2dfunet", "2dfunetlstm", "2dunet++", "kaggle"]:
                data, target = data, target[ :, 0:1, :, :]
            else:
                raise ValueError("model_mode should be ['3dswin', '2dswin', '3dunet', '2dunet']")
                # print(data, target)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    out_1 = []
                    for i, x in enumerate(range(4)):
                        temp_out = model_inferer(data.cuda()) if i==0 else torch.rot90(model_inferer(torch.rot90(data, k=i, dims=(-2, -1)).cuda()), k=-i, dims=(-2, -1))
                        out_1.append(temp_out)
                    out_1 = torch.stack(out_1, dim=0)
                    logits = out_1.mean(0)
                    # logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            loss = loss_func(logits, target)
            if args.debug:
                print(logits.shape, target.shape)
                torch.save(logits, "logits.pt")
                torch.save(target, "target.pt")
            
            
            if writer is not None:
                add_fig(writer=writer, y=target, y_pred=logits, global_step=epoch)
            logits = binary(logits, threshold=args.threshold)
            target = binary(target, threshold=args.threshold)
            val_outputs_list = decollate_batch(logits)
            # val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_labels_list = decollate_batch(target)
            # val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            acc_func.reset()
            acc_func(y_pred=val_outputs_list, y=val_labels_list)
            acc = acc_func.aggregate()[0]

            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return acc, loss.item()


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    # Tensorboard添加
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        # writer.add_hparams({
        #                     'model_mode': args.model_mode,
        #                     'loss_mode': args.loss_mode,
        #                     'effnet_mode': args.eff,
        #                     'mid': args.mid,
        #                     'lr': args.optim_lr, 
        #                     'optim_name': args.optim_name,
        #                     'reg_weight': args.reg_weight,
        #                     'momentum': args.momentum,
        #                     'roi_x': args.roi_x,
        #                     'roi_z': args.roi_z,
        #                     'dropout_rate': args.dropout_rate,
        #                     'infer_overlap': args.infer_overlap,
        #                     'lrschedule': args.lrschedule,
        #                     },
        #                   {'acc': 0})
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        epoch += 1
        
        # rank 别动。没用的，懒得改
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch+1, args.max_epochs ),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None and epoch % 2 == 0:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_lr", optimizer.param_groups[0]['lr'], epoch)
            
        b_new_best = False
        if (epoch) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            print("Start_val")
            val_avg_acc, val_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                writer=writer
            )

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                    writer.add_scalar("val_loss", val_loss, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(float(val_acc_max), float(val_avg_acc)))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename=f"model_final_{str(epoch)}.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, f"model_final_{str(epoch)}.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
