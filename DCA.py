 #模型结构上来看Variational Autoencoders 和Autoencoders 的区别主要在Coding Layer：VAE希望在模型输入和输出尽量相同的时候，同时编码层拟合一个正态分布（例如标准正态分布）。
# VAE希望编码层最终的输出向量仿佛是从一个正态分布抽样出来的，同时还和输入很像。  此时输入直接使用未屏蔽的数据来输入训练， 而测试的时候直接根据屏蔽的输入根据生成的分布来进行预测真实值   VAE就是一种拟合数据分布的gan，然后gru是根据时间序列预测输出
# from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
# from keras.utils import to_categorical
# from VAE import VAE
from Data_Process import read_datas
import scimpute
import os
from dgru import vae2
from tensorflow.keras.callbacks import EarlyStopping

def metrics_vae(y_true,y_pred):
    print('y_true: ', y_true, '\n', 'y_pred: ', y_pred)
    # print('y_true:',y_true.shape,'y_pred:',y_pred)
    omega = tf.sign(y_true)  # 0 if 0, 1 if > 0; not possibly < 0 in our data
    mse_nz = tf.reduce_mean(tf.multiply(tf.pow(y_true - y_pred, 2), omega))
    mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    return mse_nz

def vae_main():
    batch_size = 50
    intermediate_dim1 = 256
    intermediate_dim2 = 128
    epochs = 100
    num_classes = 10
    '''
    文件夹名
    1.PBMC_G949_21K
    2.GTEx_4tissues
    3.MAGIC_mouse
    4.PBMC_G949_10K
    5.PBMC_G5561
    '''
    Dataset_name='5.PBMC_G5561'
    # 读取数据
    input_matrix, gene_ids, cell_ids =read_datas()
    m, n = input_matrix.shape  # input_matrix 是sparse矩阵
    # input_matrix=input_matrix.T
    raw=input_matrix.todense()

    index=[i for i in range(raw.shape[0])]
    #聚类
    # raw=clustering(raw)

    # Dataset—process   使用Attention查找相似基因
    # Target_gene, Predict_gene, gene_to_impute ,norm_data= DataSet_process(raw)

    print("Normalization")
    # norm_data = np.log10(np.add(data, 1))
    # print(norm_data.index.tolist())
    test_cells=np.random.choice(index,int(0.1*raw.shape[0]),replace=False)
    train_cells=np.setdiff1d(index,test_cells)
    raw=pd.DataFrame(raw)
    test_cells=raw.loc[test_cells]
    train_cells=raw.loc[train_cells]
    print('train_cells : ',train_cells,'test_cells: ',test_cells)
    # print(norm_data)
    # for inputgenes in Predict_gene:
    #     print(inputgenes.tolist())
    #     print(train_cells.shape)
    #     print(norm_data.loc[train_cells,inputgenes.tolist()].values)
    # print(norm_data)
    # print(Predict_gene)
    # input_train=[norm_data.loc[train_cells,inputgenes].values for inputgenes in Predict_gene]
    # y_train=[norm_data.loc[train_cells,targetgenes].values for targetgenes in Target_gene]
    # # print(np.array(input_train).shape,np.array(y_train).shape)
    # input_test=[norm_data.loc[test_cells,inputgenes].values for inputgenes in Predict_gene]
    # y_test=[norm_data.loc[test_cells,targetgenes].values for targetgenes in Target_gene]

    # input_train=np.array(input_train)
    # input_test=np.array(input_test)
    # input_train=np.reshape(input_train,(input_train.shape[0],input_train.shape[1],input_train.shape[2],1))
    # input_test=np.reshape(input_test,(input_test.shape[0],input_test.shape[1],input_test.shape[2],1))
    history_list=[]
    history_list_train=[]
    mask_train=[]
    mask_test=[]
    #数据集屏蔽  可省略
    input_train_single = tf.nn.dropout(train_cells, 0.8)
    input_test_single = tf.nn.dropout(test_cells, 0.8)

    #加入tensor转换为numpy函数
    input_train_single.numpy()
    input_train_single.numpy()
    mask_train.append(input_train_single)
    mask_test.append(input_test_single)

    # print(input_train.shape,np.array(y_train).shape)
    # print(input_test.shape,np.array(y_test).shape)

    # input_train_single=np.reshape(input_train_single,(input_train_single.shape[0],int(input_train_single.shape[1]/50),50))
    # input_test_single=np.reshape(input_test_single,(input_test_single.shape[0],int(input_test_single.shape[1]/50),50))
    #初始化
    model=vae2(input_train_single)
    weight_path = f'/home/yangtianyu/ljs_data/DCA/checkpoint/{Dataset_name}/weight_gru'+'.ckpt'
    if os.path.exists(weight_path + '.index'):
        model.load_weights(weight_path)

    # 最优模型保存
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weight_path, save_weights_only=True, save_best_only=True,monitor='loss')      #save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
    history=model.fit(input_train_single,train_cells,validation_data=(input_test_single,test_cells),epochs=100,batch_size=32,validation_freq=1,callbacks=[cp_callback,EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')])
    # print(history.history)
    val_loss=history.history['val_loss']
    loss=history.history['loss']

    save_dir = f'/home/yangtianyu/ljs_data/DCA/loss/val_nz_loss/{Dataset_name}/'
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则自动创建
    np.savetxt(f'/home/yangtianyu/ljs_data/DCA/loss/val_nz_loss/{Dataset_name}/val_nz_loss.txt', val_loss, fmt='%s')
    
    save_dir_train = f'/home/yangtianyu/ljs_data/DCA/loss/train_nz_loss/{Dataset_name}/'
    os.makedirs(save_dir_train, exist_ok=True)  # 如果目录不存在，则自动创建
    np.savetxt(f'/home/yangtianyu/ljs_data/DCA/loss/train_nz_loss/{Dataset_name}/train_nz_loss.txt', loss,fmt='%s')
    # for j in val_loss:
    #     min=10
    #     if min>j:
    #         min=j
    # history_list.append(min)
    #
    # for m in loss:
    #     min=10
    #     if min>m:
    #         min=m
    # history_list_train.append(min)

    # sum=0
    # for i in history_list:
    #     sum+=i
    # print(history_list)
    # print(sum/input_train.shape[0])


    #保存结果
    imputeds_train=[]
    imputeds_test=[]
    weight_path = f'/home/yangtianyu/ljs_data/DCA/checkpoint/{Dataset_name}/weight_gru' + '.ckpt'
    model.load_weights(weight_path)

    # 数据集屏蔽  可省略
    # input_train_single = tf.nn.dropout(input_train_single, 0.8)
    # input_test_single = tf.nn.dropout(input_test_single, 0.8)
    # input_train_single=mask_train[0]
    # input_test_single=mask_test[0]
    # print(input_train.shape,np.array(y_train).shape)
    # print(input_test.shape,np.array(y_test).shape)

    # input_train_single = np.reshape(input_train_single,
    #                                 (input_train_single.shape[0], int(input_train_single.shape[1] / 50), 50))
    # input_test_single = np.reshape(input_test_single,
    #                                (input_test_single.shape[0], int(input_test_single.shape[1] / 50), 50))
    # session=tf.Session()
    imputeds_train.append(model.predict(input_train_single))
    imputeds_test.append(model.predict(input_test_single))

    #拼接数组
    # Imputed_train=np.hstack(imputeds_train)
    # Imputed_test=np.hstack(imputeds_test)
    print(imputeds_train[0],imputeds_test[0])
    Imputed=np.vstack((imputeds_train[0],imputeds_test[0]))

    # Y_train=np.hstack(input_train_single)
    # Y_test=np.hstack(input_test_single)
    Y=np.vstack((train_cells,test_cells))

    # mask_train=np.hstack(mask_train)
    # mask_test=np.hstack(mask_test)
    mask_input=np.vstack((mask_train[0],mask_test[0]))

    mse = tf.reduce_mean(tf.pow(imputeds_test[0] - test_cells, 2))

    omga=tf.sign(Y)
    Imputed=tf.multiply(Imputed,omga)

    omga = tf.sign(test_cells)
    Imputed_test = tf.multiply(imputeds_test[0], omga)

    mse_nz = tf.reduce_mean(tf.pow( Imputed_test - test_cells, 2))

    print('mse: ',mse,'mse_nz: ',mse_nz)

    print(type(Imputed))
    Imputed=Imputed.numpy()  #tensor  转换numpy
    Imputed_df=pd.DataFrame(data=Imputed)
    Y_df=pd.DataFrame(data=Y)
    masked_input=pd.DataFrame(data=mask_input)

    save_dir_imputed = f'/home/yangtianyu/ljs_data/DCA/imputed/{Dataset_name}/'
    os.makedirs(save_dir_imputed, exist_ok=True)  # 自动创建目录

    # 确保保存 masked_input 结果的目录存在
    save_dir_masked_input = f'/home/yangtianyu/ljs_data/DCA/masked_input/{Dataset_name}/'
    os.makedirs(save_dir_masked_input, exist_ok=True)  # 自动创建目录

    scimpute.save_hd5(Imputed_df,f'/home/yangtianyu/ljs_data/DCA/imputed/{Dataset_name}/imputation.hd5')
    scimpute.save_hd5(Y_df,f'/home/yangtianyu/ljs_data/DCA/imputed/{Dataset_name}/ground_truth.hd5')
    scimpute.save_hd5(masked_input,f'/home/yangtianyu/ljs_data/DCA/masked_input/{Dataset_name}/masked_input.hd5')
    # 最优模型保存
    # vae.summary()

    # predict=vae.predict(input_test_mask)
    # loss=metrics_vae(input_test,predict)
    # print('vaild metrics: ',loss)


if __name__== '__main__':
    vae_main()
