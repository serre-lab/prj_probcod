#!/bin/sh

python3 eval.py   --PathVAE "../simulation/IVAE_lrsvi=0.0001_lr=0.001_enc=[512, 256, 10]_it=20.pth" \
                  --PathClassifier "../simulation/CL.pth" \
                  --batch_size 1000 \
                  --verbose True \
                  --config 'config_eval.json'
                  #--NoiseType "white" "gaussian" \
                  #--white_param 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
                  #--gaussian_param 0.6 0.9