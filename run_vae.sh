#!/bin/bash

## architecture
type=VAE
hdim1=512
hdim2=256
zdim=15

activation_function=tanh
layer=fc
decoder_type='gaussian'
#beta=4
declare -a beta_list=(1)
#declare -a beta_list=(0 0.5 1 1.5 2 2.5)
#beta=1
## inference
svi_lr=1e-2
#svi_lr=1e-4
nb_it=100
#nb_it=10000
svi_optimizer=Adam

## training
lr=1e-3
nb_epoch=200
train_optimizer=Adam
seed=1
device=5

verbose=1

for beta in ${beta_list[@]}; do
  NOW=$(date +"%Y-%m-%d_%H-%M-%S")
  exp_name="${NOW}_${type}_svi_lr=${svi_lr}_lr=${lr}_beta=${beta}_nb_it=${nb_it}_[${hdim1},${hdim2},${zdim}]_af=${activation_function}_layer=${layer}_decoder=${decoder_type}"
  path="../prj_probcod_exps/$exp_name"
  rm -rf $path

  CUDA_VISIBLE_DEVICES=$device python3 train_vae.py \
      --type $type \
     --svi_lr $svi_lr \
     --nb_it $nb_it \
     --nb_epoch $nb_epoch \
     --lr $lr \
     --path $path \
     --arch $hdim1 $hdim2 \
     --z_dim $zdim \
     --activation_function $activation_function \
     --layer $layer \
     --svi_optimizer $svi_optimizer \
     --train_optimizer $train_optimizer \
     --seed $seed \
     --beta $beta \
     --decoder_type $decoder_type \
     --verbose $verbose
done

