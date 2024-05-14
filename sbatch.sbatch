#!/bin/bash
SBATCH --partition=cnGPUtest #要申请的gpu节点
SBATCH --job-name=JIN #任务名
SBATCH -o JIN.out #正确输出
SBATCH -e JIN.err #异常输出
SBATCH --gres=gpu:1 #GPU卡数
SBATCH --cpus-per-gpu=3 #每个GPU卡分配的cpu数⽬
srun env | grep CUDA #在Test%j.out中看到分配到的GPU卡的编号
srun python Tmain.py #如果是python脚本，⽤这种写法
