#!/bin/sh

device=7
type=CL
batch_size=64
epoch=100

NOW='test'
# NOW=$(date +"%m-%d-%Y_%H-%M-%S")
exp_name="${NOW}_${type}"

path="../prj_probcod_exps/$exp_name"

rm -r $path

CUDA_VISIBLE_DEVICES=$device python3 train_classifier.py \
                              --path $path\
                              --batch-size $batch_size \
                              --epochs $epoch

