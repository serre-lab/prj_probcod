#!/bin/sh

type=IVAE
#type=VAE

lr_svi=1e-2
lr=1e-3

nb_it=20

nb_epoch=100

device=7
hdim1=512
hdim2=256
zdim=5
verbose=false
activation_function=relu

# NOW=$(date +"%m-%d-%Y_%H-%M-%S")
NOW='test2'

exp_name="${NOW}_${type}_lrsvi=${lr_svi}_lr=${lr}_nb_it=${nb_it}_[${hdim1},${hdim2},${zdim}]_af=${activation_function}"


path="../prj_probcod_exps/$exp_name"

rm -r $path 

CUDA_VISIBLE_DEVICES=$device python3 train_vae.py \
    --type $type \
    --lr_svi $lr_svi \
    --nb_it $nb_it \
    --nb_epoch $nb_epoch \
    --lr $lr \
    --path $path \
    --arch $hdim1 $hdim2 \
    --z_dim $zdim \
    --activation_function $activation_function
    # --verbose $verbose \
