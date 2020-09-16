#!/bin/sh

device=7

batch_size=1000

config="config_eval.json"
PathVAE='../simulation/IVAE_lrsvi=0.0001_lr=0.001_enc=[512,256,10]_it=20.pth'
PathClassifier="../simulation/CL.pth"


CUDA_VISIBLE_DEVICES=$device python3 eval.py  \
                  --PathVAE  $PathVAE \
                  --PathClassifier  $PathClassifier\
                  --batch_size $batch_size \
                  --verbose True \
                  --config $config
