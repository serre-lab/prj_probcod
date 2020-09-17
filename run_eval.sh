#!/bin/sh

device=7

batch_size=1000

config="config_eval.json"
#PathVAE='../simulation/IVAE_lrsvi=0.0001_lr=0.001_enc=[512,256,10]_it=20.pth'
PathVAE='../prj_probcod_exps/test2_IVAE_lrsvi=1e-4_lr=1e-3_nb_it=20_[512,256,5]/model_results.pth'

PathClassifier='../prj_probcod_exps/test_CL/model_results.pth'
#PathClassifier="../simulation/CL.pth"
# NOW=$(date +"%m-%d-%Y_%H-%M-%S")

NOW='test2'

exp_name="${NOW}_EVAL"


path="../prj_probcod_exps/$exp_name"

rm -r $path


CUDA_VISIBLE_DEVICES=$device python3 eval.py  \
                  --PathVAE  $PathVAE \
                  --PathClassifier  $PathClassifier\
                  --batch_size $batch_size \
                  --verbose True \
                  --config $config \
                  --path $path