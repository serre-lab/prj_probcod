#!/bin/bash


svi_lr_eval=1e-2
nb_it_eval=500
freq_extra=25


device=4
batch_size=256
normalized_output=1

declare -a PathVAE_list=(
"../prj_probcod_exps/2020-09-22_07-03-25_IVAE_svi_lr=1e-2_lr=1e-3_beta=2.5_nb_it=20_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"

)

PathClassifier='../prj_probcod_exps/2020-09-21_08-24-05_CL'

config="config_eval.json"

for PathVAE in ${PathVAE_list[@]}; do
#NOW='test'
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
exp_name="${NOW}_EVAL_lrsvi=${svi_lr_eval}_nb_it=${nb_it_eval}"
path="../prj_probcod_exps/$exp_name"
CUDA_VISIBLE_DEVICES=$device python3 eval.py  \
                --PathVAE  $PathVAE \
                --PathClassifier  $PathClassifier\
                --batch_size $batch_size \
                --verbose True \
                --config $config \
                --path $path \
                --nb_it_eval $nb_it_eval \
                --freq_extra $freq_extra \
                --svi_lr_eval $svi_lr_eval \
                --normalized_output $normalized_output
done