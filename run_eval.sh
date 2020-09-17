#!/bin/sh


NOW='test2'

type=IVAE

lr_svi=1e-2
lr=1e-3

nb_it=20

nb_epoch=100
device=6
hdim1=512
hdim2=256
zdim=5


batch_size=256

config="config_eval.json"
#PathVAE='../simulation/IVAE_lrsvi=0.0001_lr=0.001_enc=[512,256,10]_it=20.pth'
exp_name="${NOW}_${type}_lrsvi=${lr_svi}_lr=${lr}_nb_it=${nb_it}_[${hdim1},${hdim2},${zdim}]"
PathVAE="../prj_probcod_exps/$exp_name/model_results.pth"

PathClassifier='../prj_probcod_exps/test_CL/model_results.pth'
#PathClassifier="../simulation/CL.pth"




nb_it_eval=5000
freq_extra=250
lr_svi_eval=1e-4

NOW=$(date +"%m-%d-%Y_%H-%M-%S")
exp_name="${NOW}_EVAL_${type}_lrsvi=${lr_svi_eval}"
path="../prj_probcod_exps/$exp_name"
rm -r $path

CUDA_VISIBLE_DEVICES=$device python3 eval.py  \
                  --PathVAE  $PathVAE \
                  --PathClassifier  $PathClassifier\
                  --batch_size $batch_size \
                  --verbose True \
                  --config $config \
                  --path $path \
                  --nb_it_eval $nb_it_eval \
                  --freq_extra $freq_extra \
                  --lr_svi_eval $lr_svi_eval