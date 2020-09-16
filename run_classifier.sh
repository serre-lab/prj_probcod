#!/bin/sh

python3 train_classifier.py   --path ../simulation/ \
                              --batch-size 64 \
                              --log-interval 100 \
                              --epochs 30

#python3 train_vae.py  --lr_svi 1e-4 \
#                      --nb_epoch 200 \
#                      --lr 1e-3 \
#                      --path ../simulation/ \
#                      --type IVAE \
#                      --archi 512 256 40 \
#                      --nb_it 20

#python3 train_vae.py  --lr_svi 1e-4 \
#                      --nb_epoch 200 \
#                      --lr 1e-3 \
#                      --path ../simulation/ \
#                      --type IVAE \
#                      --archi 512 256 20 \
#                      --nb_it 20

#python3 train_vae.py  --lr_svi 1e-4 \
#                      --nb_epoch 200 \
#                      --lr 1e-3 \
#                      --path ../simulation/ \
#                      --type IVAE \
#                      --archi 512 256 10 \
#                      --nb_it 20

#python3 train_vae.py  --lr_svi 1e-4 \
#                      --nb_epoch 200 \
#                      --lr 1e-3 \
#                      --path ../simulation/ \
#                      --type IVAE \
#                      --archi 512 256 5 \
#                      --nb_it 20
