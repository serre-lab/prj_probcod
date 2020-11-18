#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -J CL
#SBATCH -p gpu 
#SBATCH -o logs/%x_%A_%a_%J.out
#SBATCH -e logs/%x_%A_%a_%J.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
# SBATCH --account=carney-tserre-condo

# SBATCH -C quadrortx
# SBATCH --array=0-89

module load anaconda/3-5.2.0
source activate py36

type=CL
batch_size=64
epoch=40

#NOW=''
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
exp_name="${NOW}_${type}"


EXP_PATH="/users/azerroug/scratch"

path="${EXP_PATH}/prj_probcod_exps/$exp_name"

DATA_DIR="${EXP_PATH}/MNIST/"

rm -rf $path



python train_classifier.py \
            --path $path\
            --batch-size $batch_size \
            --epochs $epoch \
            --eval-freq 10 \
            --data_dir $DATA_DIR

