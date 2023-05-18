ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
sleep 1
nohup tensorboard --port 6006 --logdir ./runs &

