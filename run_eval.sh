#!/bin/bash


svi_lr_eval=1e-3
nb_it_eval=500
freq_extra=25

save_in_db=0

save_latent=1
device=3

save_data_reconstruction=1
# devices=(4 5 6 7 4 5) #

batch_size=256
normalized_output=1
denoising_baseline=0
per_sample_monitoring=1

declare -a PathVAE_list=(
#"../prj_probcod_exps/2020-09-25_02-37-52_IVAE_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
#"../prj_probcod_exps/2020-09-25_02-40-17_VAE_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
#"../prj_probcod_exps/2020-09-25_02-31-15_PCN_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
"../prj_probcod_exps/2020-09-25_07-17-33_PCN_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
"../prj_probcod_exps/2020-09-25_07-17-33_PCN_svi_lr=1e-2_lr=1e-3_beta=0.1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
"../prj_probcod_exps/2020-09-25_07-18-57_PCN_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
"../prj_probcod_exps/2020-09-25_07-18-57_PCN_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
"../prj_probcod_exps/2020-09-25_08-37-46_PCN_svi_lr=1e-2_lr=1e-3_beta=5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
"../prj_probcod_exps/2020-09-25_08-37-46_PCN_svi_lr=1e-2_lr=1e-3_beta=10_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian"
)

DATA_DIR='../DataSet/MNIST/'

PathClassifier='../prj_probcod_exps/2020-09-25_09-17-58_CL'

config="config_eval.json"

for idx in {4..5}; do
#NOW='test'
device=${devices[$idx]}
PathVAE=${PathVAE_list[$idx]}

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

exp_name="${NOW}_EVAL_lrsvi=${svi_lr_eval}_nb_it=${nb_it_eval}"
path="../prj_probcod_exps/$exp_name"
CUDA_VISIBLE_DEVICES=$device \
nohup python3 eval.py  \
                --PathVAE  $PathVAE \
                --PathClassifier  $PathClassifier\
                --batch_size $batch_size \
                --verbose True \
                --config $config \
                --path $path \
                --nb_it_eval $nb_it_eval \
                --freq_extra $freq_extra \
                --svi_lr_eval $svi_lr_eval \
                --normalized_output $normalized_output \
                --save_in_db $save_in_db \
                --save_latent $save_latent \
                --denoising_baseline $denoising_baseline \
                --per_sample_monitoring $per_sample_monitoring
                --data_dir $DATA_DIR \
                > logs/EVAL_${idx}.log 2>&1 &
done

