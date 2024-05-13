#!/bin/bash
SBATCH --job-name=JIN #任务名
SBATCH --partition=cnGPUtest #要申请的gpu节点
SBATCH -o JIN.out #正确输出
SBATCH -e JIN.err #异常输出
SBATCH --gres=gpu:1 #GPU卡数
SBATCH --cpus-per-gpu=3 #每个GPU卡分配的cpu数⽬
srun env | grep CUDA #在Test%j.out中看到分配到的GPU卡的编号
srun python Tmain.py #如果是python脚本，⽤这种写法











#!/bin/bash
# 指定作业名称，以便在队列预览时识别
#SBATCH --job-name=JIN
# 指定作业的输出文件，出错信息文件
#SBATCH -o JIN.out
#SBATCH -e JIN.err
# 作业完成或失败时的邮件通知设置（暂未支持）
#SBATCH --mail-type=END,FAIL # 通知类型，作业完成及失败
#SBATCH --mail-user=queziaa@163.com
# 定义作业预计运行时长，d-hh:mm:ss
#SBATCH --time 15:00:00
# 为GPU作业指定partition
#SBATCH --partition=cnGPUtest
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
# 任务数
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=3
#srun ~/cuda_hello 