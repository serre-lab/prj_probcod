#!/bin/sh


python3 train_vae.py --lr_svi 1e-4 --nb_epoch 1 --lr 1e-3 --path ../simulation/ --type IVAE

