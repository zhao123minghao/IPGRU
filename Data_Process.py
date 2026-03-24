import os
import numpy as np
import pandas as pd
import math
import scimpute
from scipy.sparse import csr_matrix

#数据地址
'''
1. /mnt/qwe2/ljs_data/Single_Cell_Data/Dataset/PBMC_G949_21K/ground_truth/pbmc.g949_c21k.hd5
2. /mnt/qwe2/ljs_data/Single_Cell_Data/Dataset/GTEx_4tissues/ground_truth/gtex_v7.4TISSUES.count.cell_row.hd5
3. /mnt/qwe2/ljs_data/Single_Cell_Data/Dataset/MAGIC_mouse/ground_truth/MBM.MAGIC.9k.B.hd5
4. /mnt/qwe2/ljs_data/Single_Cell_Data/Dataset/PBMC_G949_10K/ground_truth/pbmc.g949_c10k.hd5
5. /mnt/qwe2/ljs_data/Single_Cell_Data/Dataset/PBMC_G5561/input/pbmc.g5561_c54k.hd5
/home/yangtianyu/ljs_data
/home/yangtianyu/ljs_data/AutoImpute/imputed/1.PBMC_G949_21K/imputation.hd5
'''
def read_datas():
    print('读取hd5格式数据为dataframe格式：..')
    input_df = scimpute.read_data_into_cell_row('/home/yangtianyu/ljs_data/AutoImpute/imputed/1.PBMC_G949_21K/imputation.hd5','cell_row')
    # input_df=input_df.T

    print('Data Transformation')  #取对数 log10（X+1）标准化
    input_df=scimpute.df_transformation(input_df.transpose(),transformation='log10').transpose()

    print('将pandas矩阵转换为sparse矩阵')
    input_matrix = csr_matrix(input_df)
    gene_ids=input_df.columns
    cell_ids=input_df.index

    print('name_input:','example.msk90')
    _=pd.DataFrame(data=input_matrix[:20,:4].todense(),index=cell_ids[:20],columns=gene_ids[:4])
    print('input_df:\n',_,'\n')
    m,n=input_matrix.shape
    print('input_matrix:{} cells , {} genes\n'.format(m,n))
    return input_matrix,gene_ids,cell_ids

