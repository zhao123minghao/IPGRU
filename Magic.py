import pandas as pd
import tensorflow as tf
import numpy as np
import scimpute
'''
文件夹名
1.PBMC_G949_21K
2.GTEx_4tissues
3.MAGIC_mouse
4.PBMC_G949_10K
5.PBMC_G5561
/mnt/qwe2/ljs_data/csv_data/
'''
dataset_name='5.PBMC_G5561'
raw=pd.read_csv(f'/home/yangtianyu/ljs_data/MAGIC/{dataset_name}.csv')

magic = pd.read_csv(f'/home/yangtianyu/ljs_data/MAGIC/{dataset_name}'+'_magic.csv')

raw=raw.iloc[:,1:]
magic=magic.iloc[:,1:]

raw=scimpute.df_transformation(raw.transpose(),transformation='log10').transpose()
magic=scimpute.df_transformation(magic.transpose(),transformation='log10').transpose()

raw=np.array(raw)

magic=np.array(magic)
print('raw.shape: ',raw.shape,'magic.shape',magic.shape)
print('raw ',raw,'magic',magic)

mse=tf.reduce_mean(tf.pow(raw-magic,2))

omng=tf.sign(raw)

magic_nz=tf.multiply(omng,magic)

mse_nz=tf.reduce_mean(tf.pow(raw-magic_nz,2))

print('mse: ',mse,'mse_nz: ',mse_nz)

