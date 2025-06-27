#!/bin/bash

# 首次使用此脚本需要赋予权限
# chmod +x ./train_script.sh
# 加载 conda 初始化脚本
# source /home/huxian/miniconda3/etc/profile.d/conda.sh

# 激活指定的 conda 环境
# conda activate vjepa2


# 生成当前时间的时间戳
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# 使用时间戳命名日志文件
logfile="${timestamp}_train.log"

# 环境变量和命令
command='PYTHONPATH=~/project/vjepa2 CUDA_VISIBLE_DEVICES=7 python ./app/main.py --fname configs/train/vitg16/tr2-256px-8f.yaml --device cuda:7'

# 使用nohup命令运行脚本并将输出重定向到日志文件
nohup bash -c "$command" > "$logfile" 2>&1 &

# 提示日志文件的位置
echo "Training script is running in the background. Check the log file: $logfile"
