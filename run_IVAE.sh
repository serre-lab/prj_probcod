#!/bin/bash

## architecture
# type=IVAE
type=SVI
hdim1=512
hdim2=256
zdim=15

activation_function=tanh
layer=fc
decoder_type='gaussian'
#beta=4
# beta_list=(1)
beta_list=(0 0.5 1 1.5 2 2.5)

#beta=1
## inference
svi_lr=1e-2
#svi_lr=1e-4
nb_it=10
#nb_it=10000
svi_optimizer=Adam

## training
lr=1e-3
nb_epoch=3
train_optimizer=Adam
seed=1

device=(0 1 2 3 4)
verbose=1

var_init=0.1

DATA_DIR='../DataSet/MNIST/'

path_db="../probcod_dbs/"

# for beta in ${beta_list[@]}; do
for i in {0..4}; do

  beta=${beta_list[$i]}
  device=$i

  NOW=$(date +"%Y-%m-%d_%H-%M-%S")
  exp_name="${NOW}_${type}_svi_lr=${svi_lr}_lr=${lr}_beta=${beta}_nb_it=${nb_it}_[${hdim1},${hdim2},${zdim}]_af=${activation_function}_layer=${layer}_decoder=${decoder_type}"
  path="../prj_probcod_exps/$exp_name"
  rm -rf $path

  CUDA_VISIBLE_DEVICES=$device \
  nohup \
  python3 train_vae.py \
      --exp_name $exp_name \
      --path_db $path_db \
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
      --var_init $var_init \
      --svi_optimizer $svi_optimizer \
      --train_optimizer $train_optimizer \
      --seed $seed \
      --beta $beta \
      --decoder_type $decoder_type \
      --verbose $verbose \
      --data_dir $DATA_DIR \
      > logs/${exp_name}.log 2>&1 &

done