ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9

nohup tensorboard --port 6006 --logdir ./runs &

