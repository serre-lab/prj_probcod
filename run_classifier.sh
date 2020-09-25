#!/bin/sh

device=4
type=CL
batch_size=64
epoch=40

#NOW=''
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
exp_name="${NOW}_${type}"

path="../prj_probcod_exps/$exp_name"

DATA_DIR='../DataSet/MNIST/'

rm -rf $path



CUDA_VISIBLE_DEVICES=$device python3 train_classifier.py \
                              --path $path\
                              --batch-size $batch_size \
                              --epochs $epoch \
                              --eval-freq 10 \
                              --data_dir $DATA_DIR

