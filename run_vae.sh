#!/bin/sh

type=IVAE

lr_svi=1e-4
lr=1e-3

nb_it=20

nb_epoch=200

device=0

verbose=false

# NOW=$(date +"%m-%d-%Y_%H-%M-%S")
NOW='test'

exp_name="${NOW}_${type}_lrsvi=${lr_svi}_lr=${lr}_nb_it=${nb_it}"


path="../prj_probcod_exps/$exp_name"

rm -r $path 

CUDA_VISIBLE_DEVICES=$device python3 train_vae.py \
    --type $type \
    --lr_svi $lr_svi \
    --nb_it $nb_it \
    --nb_epoch $nb_epoch \
    --lr $lr \
    --path $path \
    # --verbose $verbose \
