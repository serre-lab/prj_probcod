#!/bin/sh

type=CL
device=4
batch_size=1024
normalized_output=1

# PathClassifier='../prj_probcod_exps/2020-09-21_08-24-05_CL'
# PathClassifier='../prj_probcod_exps/2020-09-21_08-24-05_CL'
PathClassifier='../prj_probcod_exps/2020-09-24_10-49-23_CL'


config="config_eval.json"


NOW=$(date +"%Y-%m-%d_%H-%M-%S")
exp_name="${NOW}_EVAL_${type}"

path="../prj_probcod_exps/$exp_name"

rm -r $path

CUDA_VISIBLE_DEVICES=$device python3 eval_classifier.py \
                              --path $path\
                              --batch_size $batch_size \
                              --PathClassifier $PathClassifier \
                              --config $config \
                              --normalized_output $normalized_output

