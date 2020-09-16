#!/bin/sh

python3 eval.py   --NoiseType "white" "gaussian" \
                  --PathVAE "../simulation/IVAE_lrsvi=0.0001_lr=0.001_enc=[512, 256, 10]_it=20.pth" \
                  --PathClassifier "../simulation/CL.pth"