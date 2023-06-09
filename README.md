# 运行代码

## log
现在不需要指定logdir。 
自动生成模板：
```python
f"{args.roi_x}_{args.model_mode}_{args.eff}_{args.roi_z}_mid{args.mid}_{args.optim_name}_{time.strftime('%b-%d-%H-%M', time.gmtime(time.time()))}"
```
所有的log都在runs下面

## Tensorboard
tensorboard打开代码看这个: [autodl tensorboard](https://www.autodl.com/docs/tensorboard/)

```bash
nohup tensorboard --port 6007 --logdir /root/autodl-tmp/SwinUNETR/runs &
```
加上nohup 可以后台运行，就算关掉了自己的电脑也能运行。训练同理，前面加`nohup`后面加`&`

## 训练参数

- cache_rate是用来控制缓存数据的。如果被kill了，就加上`--use_normal_dataset`
- num_samples就当batch_siz
- `--workers 0` 不要删。
- `val_every` 是多少epoch validation一次。每次validation都会保存一次模型

```bash
python main.py --model_mode 2dfunetlstm --roi_x 512 --roi_y 512 --num_samples 20 --save_checkpoint --workers 0 --val_every 10 --lrschedule cosine_anneal --optim_lr 9e-4 --roi_z 22 --cache_rate 1.0 --eff b5 --optim_name sgd --mid 18  --max_epochs 2000 
```

上次训练的代码
```bash
python main.py --model_mode 2dfunetlstm --eff b6 --roi_x 384 --roi_y 384 --num_samples 10 --save_checkpoint  --val_every 40 --lrschedule warmup_cosine --optim_lr 4e-4  --cache_rate 1.0 --optim_name adamw  --max_epochs 3000 --normal --workers 10 --mid 19 --roi_z 22
```
