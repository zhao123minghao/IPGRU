
import scimpute
import tensorflow as tf
'''
   文件夹名
    1.PBMC_G949_21K
    2.GTEx_4tissues
    3.MAGIC_mouse
    4.PBMC_G949_10K
    5.PBMC_G5561
'''
def analysising():
    print('读取hd5格式数据为dataframe格式：..')
    vaild_imputed_df = scimpute.read_data_into_cell_row('/home/yangtianyu/ljs_data/AutoImpute2/imputed/5.PBMC_G5561/imputation.hd5','cell_row')
    vaild_truth_df = scimpute.read_data_into_cell_row('/home/yangtianyu/ljs_data/AutoImpute2/imputed/5.PBMC_G5561/ground_truth.hd5', 'cell_row')
    # input_df=input_df.T
    # df_tmp = input_df.copy()
    # input_df = np.power(10, df_tmp) - 1
    # print('Data Transformation')  #取对数 log10（X+1）标准化
    # input_df=scimpute.df_transformation(input_df.transpose(),transformation='log10').transpose()
    x=vaild_imputed_df.shape[0]
    m = vaild_truth_df.shape[0]
    x1=int(x*0.1)
    m1 = int(m * 0.1)
    vaild_imputed_df = vaild_imputed_df.values
    vaild_truth_df = vaild_truth_df.values
    vaild_imputed_df=vaild_imputed_df[x-x1-1:,:]
    vaild_truth_df=vaild_truth_df[m-m1-1:,:]
    # vaild_imputed_df = vaild_imputed_df.values
    # vaild_truth_df=vaild_truth_df.values
    # print('y_true: ', y_true, '\n', 'y_pred: ', y_pred)
    # print('y_true:',y_true.shape,'y_pred:',y_pred)
    omega = tf.sign(vaild_truth_df)  # 0 if 0, 1 if > 0; not possibly < 0 in our data
    print('omega: ', omega.numpy())
    mse_nz = tf.reduce_mean(tf.multiply(tf.pow(vaild_truth_df-vaild_imputed_df,2), omega))
    mse = tf.reduce_mean(tf.pow(vaild_truth_df - vaild_imputed_df, 2))
    print('mse_nz: ',mse_nz,'mse: ',mse)
    # print('将pandas矩阵转换为sparse矩阵')
    # input_matrix = csr_matrix(input_df)
    # gene_ids=input_df.columns
    # cell_ids=input_df.index
    # print('name_input:','example.msk90')
    # _=pd.DataFrame(data=input_matrix[:20,:4].todense(),index=cell_ids[:20],columns=gene_ids[:4])
    # print('input_df:\n',_,'\n')
    # m,n=input_matrix.shape
    # print('input_matrix:{} cells , {} genes\n'.format(m,n))
    # input_matrix=input_matrix.todense()
    # print(input_matrix)
    # return input_matrix,gene_ids,cell_ids
if __name__=='__main__':
    analysising()