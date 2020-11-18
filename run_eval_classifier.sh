#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -J CL_eval
#SBATCH -p gpu 
#SBATCH -o logs/%x_%A_%a_%J.out
#SBATCH -e logs/%x_%A_%a_%J.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=carney-tserre-condo
# SBATCH --array=0-14

module load anaconda/3-5.2.0
source activate py36

EXP_PATH="/users/azerroug/scratch"


type=CL
device=4
batch_size=1024
normalized_output=1
save_in_db=1
save_latent=1

## normal classifier
# PathClassifier="${EXP_PATH}/prj_probcod_exps/2020-09-27_18-35-43_CL"
# PathClassifier="${EXP_PATH}/prj_probcod_exps/2020-10-14_09-49-01_CL"
PathClassifier="${EXP_PATH}/prj_probcod_exps/2020-11-18_13-12-05_CL"

zca_whiten=0


## zca classifier
# PathClassifier="${EXP_PATH}/prj_probcod_exps/2020-10-13_04-10-26_CL"
# PathClassifier="${EXP_PATH}/prj_probcod_exps/2020-10-14_05-03-33_CL"
# PathClassifier="${EXP_PATH}/prj_probcod_exps/2020-10-14_09-02-53_CL"
# zca_whiten=1

path_db="db_EVAL_2.csv"

config="config_eval.json"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
exp_name="${NOW}_EVAL_${type}"


DATA_DIR="${EXP_PATH}/MNIST/"

path="${EXP_PATH}/prj_probcod_exps/$exp_name"

rm -r $path

# CUDA_VISIBLE_DEVICES=$device 
python3 eval_classifier.py \
                            --path $path \
                            --batch_size $batch_size \
                            --PathClassifier $PathClassifier \
                            --config $config \
                            --normalized_output $normalized_output\
                            --save_in_db $save_in_db \
                            --save_latent $save_latent \
                            --path_db $path_db \
                            --data_dir $DATA_DIR \
                            --zca_whiten $zca_whiten
