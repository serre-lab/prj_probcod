#!/bin/bash


module load anaconda/3-5.2.0
source activate py36

python group_dbs.py
db_name='db_EVAL_zca'
cp ${db_name}.csv /cifs/data/tserre_lrs/projects/prj_predcoding/${db_name}.csv

sed -i 's/\/users\/azerroug\/scratch/\/media\/data_cifs_lrs\/projects\/prj_predcoding/' /cifs/data/tserre_lrs/projects/prj_predcoding/${db_name}.csv
sed -i 's/\/users\/azerroug\/scratch/\/media\/data_cifs_lrs\/projects\/prj_predcoding/' /cifs/data/tserre_lrs/projects/prj_predcoding/${db_name}.csv
sed -i 's/\/users\/azerroug\/scratch/\/media\/data_cifs_lrs\/projects\/prj_predcoding/' /cifs/data/tserre_lrs/projects/prj_predcoding/${db_name}.csv

# cp -r ../scratch/prj_probcod_exps/ /cifs/data/tserre_lrs/projects/prj_predcoding/
# cp -r ../scratch/prj_probcod_exps/2020-10-01_05-* /cifs/data/tserre_lrs/projects/prj_predcoding/prj_probcod_exps

cp -r ../scratch/prj_probcod_exps/2020-10-13* /cifs/data/tserre_lrs/projects/prj_predcoding/prj_probcod_exps
