import os
import pandas as pd

path = '../scratch/prj_probcod/db_zca/'
dbs = os.listdir(path)

df = pd.DataFrame()

for db in dbs:
    df = df.append(pd.read_csv(path+db, index_col=0), ignore_index=True)

df = df.append(pd.read_csv('db_EVAL_2.csv', index_col=0), ignore_index=True)

df.to_csv('db_EVAL_zca.csv')

