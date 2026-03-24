
import pandas as pd
import numpy as np
import tensorflow as tf
# from keras.utils import to_categorical
# from VAE import VAE
from Data_Process import read_datas
import scimpute
import os
from DataSet_Process import DataSet_process
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
    Dataset_name='3.MAGIC_mouse'
    # 读取数据
    input_matrix, gene_ids, cell_ids =read_datas()
    m, n = input_matrix.shape  # input_matrix 是sparse矩阵
    # input_matrix=input_matrix.T
    raw=input_matrix.todense()

    #聚类
    # raw=clustering(raw)

    # Dataset—process   使用Attention查找相似基因
    Target_gene, Predict_gene, gene_to_impute ,norm_data= DataSet_process(raw)

    print("Normalization")
    # norm_data = np.log10(np.add(data, 1))
    # print(norm_data.index.tolist())
    test_cells=np.random.choice(norm_data.index,int(0.1*norm_data.shape[0]),replace=False)
    train_cells=np.setdiff1d(norm_data.index,test_cells)
    # print(norm_data)
    # for inputgenes in Predict_gene:
    #     print(inputgenes.tolist())
    #     print(train_cells.shape)
    #     print(norm_data.loc[train_cells,inputgenes.tolist()].values)
    print(norm_data)
    print(Predict_gene)
    input_train=[norm_data.loc[train_cells,inputgenes].values for inputgenes in Predict_gene]
    y_train=[norm_data.loc[train_cells,targetgenes].values for targetgenes in Target_gene]
    # print(np.array(input_train).shape,np.array(y_train).shape)
    input_test=[norm_data.loc[test_cells,inputgenes].values for inputgenes in Predict_gene]
    y_test=[norm_data.loc[test_cells,targetgenes].values for targetgenes in Target_gene]

    input_train=np.array(input_train)
    input_test=np.array(input_test)
    # input_train=np.reshape(input_train,(input_train.shape[0],input_train.shape[1],input_train.shape[2],1))
    # input_test=np.reshape(input_test,(input_test.shape[0],input_test.shape[1],input_test.shape[2],1))
    history_list=[]
    history_list_train=[]
    print('Predict_gene: ',len(Predict_gene))
    k=0
    mask_train=[]
    mask_test=[]
    for i in range(len(Predict_gene)):
        input_train_single=input_train[i]
        input_test_single=input_test[i]
        y_train_single=y_train[i]
        y_test_single=y_test[i]


        # 这里引入新的屏蔽策略，输入 input_train_single ，输出是置0后的结果，输出与input_train_single同维度
        # 替换掉下面两行代码
        #数据集屏蔽  可省略
        input_train_single = tf.nn.dropout(input_train_single, 0.8)
        input_test_single = tf.nn.dropout(input_test_single, 0.8)

        #加入tensor转换为numpy函数
        input_train_single.numpy()
        input_train_single.numpy()
        mask_train.append(input_train_single)
        mask_test.append(input_test_single)

        # print(input_train.shape,np.array(y_train).shape)
        # print(input_test.shape,np.array(y_test).shape)

        input_train_single=np.reshape(input_train_single,(input_train_single.shape[0],int(input_train_single.shape[1]/50),50))
        input_test_single=np.reshape(input_test_single,(input_test_single.shape[0],int(input_test_single.shape[1]/50),50))
        #初始化
        model=vae2(input_train_single)
        k=k+1
        weight_path = f'/home/yangtianyu/ljs_data/AutoImpute2/checkpoint/{Dataset_name}/weight_gru{k}'+'.ckpt'
        if os.path.exists(weight_path + '.index'):
            model.load_weights(weight_path)

        # 最优模型保存
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weight_path, save_weights_only=True, save_best_only=True,monitor='loss')      #save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
        history=model.fit(input_train_single,y_train_single,validation_data=(input_test_single,y_test_single),epochs=50,batch_size=32,validation_freq=1,callbacks=[cp_callback,EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')])
        # print(history.history)
        val_loss=history.history['val_loss']
        loss=history.history['loss']

        save_dir = f'/home/yangtianyu/ljs_data/AutoImpute2/loss/val_nz_loss/{Dataset_name}/'
        os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则自动创建
        np.savetxt(f'/home/yangtianyu/ljs_data/AutoImpute2/loss/val_nz_loss/{Dataset_name}/val_nz_{k}_loss.txt', val_loss, fmt='%s')
        # 创建目录
        save_dir_train = f'/home/yangtianyu/ljs_data/AutoImpute2/loss/train_nz_loss/{Dataset_name}/'
        os.makedirs(save_dir_train, exist_ok=True)  # 如果目录不存在，则自动创建
        np.savetxt(f'/home/yangtianyu/ljs_data/AutoImpute2/loss/train_nz_loss/{Dataset_name}/train_nz_{k}_loss.txt', loss,fmt='%s')
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
    n=0
    imputeds_train=[]
    imputeds_test=[]
    for i in range(len(Predict_gene)):
        n=n+1
        weight_path = f'/home/yangtianyu/ljs_data/AutoImpute2/checkpoint/{Dataset_name}/weight_gru{n}' + '.ckpt'
        model.load_weights(weight_path)
        input_train_single = input_train[i]
        input_test_single = input_test[i]
        y_train_single = y_train[i]
        y_test_single = y_test[i]

        # 数据集屏蔽  可省略
        # input_train_single = tf.nn.dropout(input_train_single, 0.8)
        # input_test_single = tf.nn.dropout(input_test_single, 0.8)
        input_train_single=mask_train[i]
        input_test_single=mask_test[i]
        # print(input_train.shape,np.array(y_train).shape)
        # print(input_test.shape,np.array(y_test).shape)

        input_train_single = np.reshape(input_train_single,
                                        (input_train_single.shape[0], int(input_train_single.shape[1] / 50), 50))
        input_test_single = np.reshape(input_test_single,
                                      (input_test_single.shape[0], int(input_test_single.shape[1] / 50), 50))
        # session=tf.Session()
        imputeds_train.append(model.predict(input_train_single))
        imputeds_test.append(model.predict(input_test_single))

    #拼接数组
    Imputed_train=np.hstack(imputeds_train)
    Imputed_test=np.hstack(imputeds_test)
    Imputed=np.vstack((Imputed_train,Imputed_test))

    Y_train=np.hstack(y_train)
    Y_test=np.hstack(y_test)
    Y=np.vstack((Y_train,Y_test))

    mask_train=np.hstack(mask_train)
    mask_test=np.hstack(mask_test)
    mask_input=np.vstack((mask_train,mask_test))

    mse = tf.reduce_mean(tf.pow(Imputed - Y, 2))

    omga=tf.sign(Y)
    Imputed=tf.multiply(Imputed,omga)

    mse_nz = tf.reduce_mean(tf.pow( Imputed- Y, 2))

    print('mse: ',mse,'mse_nz: ',mse_nz)

    print(type(Imputed))
    Imputed=Imputed.numpy()  #tensor  转换numpy
    Imputed_df=pd.DataFrame(data=Imputed)
    Y_df=pd.DataFrame(data=Y)
    masked_input=pd.DataFrame(data=mask_input)

    save_dir_imputed = f'/home/yangtianyu/ljs_data/AutoImpute2/imputed/{Dataset_name}/'
    os.makedirs(save_dir_imputed, exist_ok=True)  # 自动创建目录

    # 确保保存 masked_input 结果的目录存在
    save_dir_masked_input = f'/home/yangtianyu/ljs_data/AutoImpute2/masked_input/{Dataset_name}/'
    os.makedirs(save_dir_masked_input, exist_ok=True)  # 自动创建目录

    scimpute.save_hd5(Imputed_df,f'/home/yangtianyu/ljs_data/AutoImpute2/imputed/{Dataset_name}/imputation.hd5')
    scimpute.save_hd5(Y_df,f'/home/yangtianyu/ljs_data/AutoImpute2/imputed/{Dataset_name}/ground_truth.hd5')
    scimpute.save_hd5(masked_input,f'/home/yangtianyu/ljs_data/AutoImpute2/masked_input/{Dataset_name}/masked_input.hd5')
    # 最优模型保存
    # vae.summary()

    # predict=vae.predict(input_test_mask)
    # loss=metrics_vae(input_test,predict)
    # print('vaild metrics: ',loss)


if __name__== '__main__':
    vae_main()
