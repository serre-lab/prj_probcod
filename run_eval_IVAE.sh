#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -J IVAE_eval
#SBATCH -p gpu 
#SBATCH -o logs/%x_%A_%a_%J.out
#SBATCH -e logs/%x_%A_%a_%J.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-74
#SBATCH --account=carney-tserre-condo

# SBATCH -C quadrortx

module load anaconda/3-5.2.0
source activate py36


svi_lr_eval=1e-2
nb_it_eval=500
freq_extra=25


device=4
batch_size=256
normalized_output=1
per_sample_monitoring=0

PathVAE_list=(\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-35_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=1"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=0"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=10"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=11"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=12"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=13"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=14"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=2"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=3"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=4"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=5"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=6"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=7"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=8"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_06-58-36_SVI_svi_lr=1e-2_lr=1e-3_beta=0_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=9"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-40-22_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=0"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-40-22_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=2"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-40-56_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=1"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-41-21_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=8"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-41-40_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=5"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-41-52_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=7"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-42-00_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=3"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-42-08_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=9"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-42-24_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=4"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-48-24_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=13"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-48-27_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=11"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-49-20_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=14"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-52-53_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=6"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_07-53-21_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=10"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-03-04_SVI_svi_lr=1e-2_lr=1e-3_beta=0.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=12"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-22-34_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=2"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-22-52_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=1"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-23-22_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=0"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-24-17_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=7"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-24-42_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=5"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-24-45_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=8"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-07-50_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=12"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-24-59_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=9"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-25-27_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=3"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-25-40_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=4"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-38-26_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=11"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-38-49_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=13"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-39-41_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=14"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-47-02_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=6"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_08-48-12_SVI_svi_lr=1e-2_lr=1e-3_beta=1_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=10"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-04-15_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=2"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-05-16_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=1"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-05-21_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=0"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-07-00_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=8"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-07-39_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=7"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-07-48_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=5"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_11-03-51_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=12"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-08-05_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=9"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-08-43_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=4"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-08-47_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=3"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-28-28_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=11"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-28-52_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=13"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-30-34_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=14"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-40-52_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=6"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-42-12_SVI_svi_lr=1e-2_lr=1e-3_beta=1.5_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=10"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-45-58_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=2"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-47-24_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=0"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-47-27_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=1"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-49-29_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=8"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-50-44_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=7"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-50-57_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=5"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-51-55_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=9"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-51-56_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=4"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_09-52-27_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=3"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_11-02-51_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=10"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_11-02-51_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=11"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_11-02-51_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=12"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_11-02-51_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=13"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_11-02-51_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=14"\
    "/users/azerroug/scratch/prj_probcod_exps/2020-11-18_11-02-51_SVI_svi_lr=1e-2_lr=1e-3_beta=2_nb_it=100_[512,256,15]_af=tanh_layer=fc_decoder=gaussian_varinit=0.001_seed=6"\
)

### normal
# PathClassifier='/users/azerroug/scratch/prj_probcod_exps/2020-09-27_18-35-43_CL'
# zca_whiten=0

### zca
# PathClassifier='/users/azerroug/scratch/prj_probcod_exps/2020-10-13_04-10-26_CL'
# PathClassifier='/users/azerroug/scratch/prj_probcod_exps/2020-10-14_09-02-53_CL'
PathClassifier='/users/azerroug/scratch/prj_probcod_exps/2020-11-18_13-12-05_CL'
# zca_whiten=1



config="config_eval.json"
# config="config_eval.json"

EXP_PATH="/users/azerroug/scratch"

DATA_DIR="${EXP_PATH}/MNIST/"


# for PathVAE in ${PathVAE_list[@]}; do
# NOW='test'
NOW=$(date +"%Y-%m-%d_%H-%M-%S")

exp_name="${NOW}_EVAL_lrsvi=${svi_lr_eval}_nb_it=${nb_it_eval}_job=$SLURM_ARRAY_TASK_ID"

path="${EXP_PATH}/prj_probcod_eval_exps/$exp_name"

PathVAE=${PathVAE_list[$SLURM_ARRAY_TASK_ID]}

# CUDA_VISIBLE_DEVICES=$device \
python eval.py  \
                --PathVAE  $PathVAE \
                --PathClassifier  $PathClassifier\
                --batch_size $batch_size \
                --verbose True \
                --config $config \
                --path $path \
                --nb_it_eval $nb_it_eval \
                --freq_extra $freq_extra \
                --svi_lr_eval $svi_lr_eval \
                --normalized_output $normalized_output \
                --per_sample_monitoring $per_sample_monitoring \
                --data_dir $DATA_DIR \
                # --zca_whiten $zca_whiten
                

# done