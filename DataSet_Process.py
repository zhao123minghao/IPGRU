import numpy as np
import pandas as pd
import Data_Process

#传入数据是numpy形式  细胞/基因矩阵   分离目标基因及预测基因

def inspect_data(data):  # 数据处理函数
    # Check if there area any duplicated cell/gene labels

    if sum(data.index.duplicated()):  # dataframe.index  获取数据的行索引标签，通过.duplicated进行行索引标签统计
        print("ERROR: duplicated cell labels. Please provide unique cell labels.")  # 鉴定行索引细胞是否重复
        exit(1)

    if sum(data.columns.duplicated()):  # dataframe.columns  获取数据的列索引标签，通过.duplicated进行列索引标签统计
        print("ERROR: duplicated gene labels. Please provide unique gene labels.")  # 鉴定列索引 基因是否重复
        exit(1)

    max_value = np.max(data.values)  # dataframe.values转换为numpy形式,np.max()求序列的最值
    if max_value < 0:  # 当细胞/基因dataframe的最大值不大于10的时候,将报错
        print("ERROR: max value = {}. Is your data log-transformed? Please provide raw counts"
              .format(max_value))
        exit(1)

    print("Input dataset is {} cells (rows) and {} genes (columns)"  # 输入矩阵的行列数
          .format(*data.shape))
    print("First 3 rows and columns:")
    print(data.iloc[:3, :3])  # iloc[   :  ,  : ]    前面的冒号就是取行数，后面的冒号是取列数
#ruanshengnet

#此处先假设输出维度为100，用500个输入基因预测这一百个基因   out_dim=100
def filter_genes(gene_metric, threshold):# assumes gene_metric is sorted
        # if not str(NN_lim).isdigit():   #str().isdigit()若是字符串只包含数字则返回true,否则返回false
    # NN_lim = (gene_metric > threshold).sum()        #将基因的列的(方差)/(均值+1)>最小阀值的列统计起来,视为插补基因总个数
    NN_lim = (gene_metric>0).sum()
    print('NN_lim, ',NN_lim)
    n_subsets = int(NN_lim / 300)   #np.ceil(nadrray)数据向上取整,np.floor(ndarray)数据向下取整      总的基因数/每个神经网络输出的神经元数  即为要划分的子神经网络个数    如果把全部的基因数放入一个神经网络可能会导致性能不好,所以把全部基因分开成几组,将这几组基因放入每个子神经网络进行训练.得到512是子神经网络神经元权衡最佳的数目

    genes_to_impute = gene_metric.index[:n_subsets*300]      #通过series.index[0:组数*每组个数]切片获取到Series的实际长度的一个range(0,1,...)数组,即为要插补的基因  从0,1,2  计数直到最后一个列数的range

    # rest = 100 - (len(genes_to_impute) % 100) #通过该式得到最后一组不全的基因列还缺少多少基因
    # drop=[]
    # for i in range(len(genes_to_impute)-rest,len(genes_to_impute)+1):
    #     drop.append(i)
    # if rest > 0:                    #若最后一组不全
    #     genes_to_impute=genes_to_impute.drop(genes_to_impute.columns[drop],axis=1,inplace=True)
    #     # fill_genes = np.random.choice(gene_metric.index, rest) #通过np.random.choice()  可以从.index一维Series存在的中选取rest个数,并组成一个数组
    #     # genes_to_impute = np.concatenate([genes_to_impute, fill_genes])     #通过np.concatenate(),可以将两个数组进行行拼接 将已存在的基因列填充到现有的列中还未满的最后一组基因组 使其成为满的一组基因列放入子神经网络训练
    #
    # print("{} genes selected for imputation".format(len(genes_to_impute)))  #将一共len()个基因被选择当作特征进行训练,预测特征插补值

    return genes_to_impute                                                  #返回一个一维数组:基因列序号数组


def get_distance_matrix(raw):
    VMR = raw.std() / raw.mean()  # 得到每列基因的VMR的一维数组
    VMR[np.isinf(VMR)] = 0  # 通过np.isinf(array)检测一个数组中的值是否是无穷大  是否结果通过布尔数组返回  将结果为是的列基因序号通过VMR[]置无穷大为0
    # print('raw: ',raw)
    # if n_pred is None:  # 若是为空  默认将基因全部选择
    potential_pred = raw.columns  # 获得VMR>0的列索引序号   将全部的VMR>0的列号作为潜在预测基因
    # else:  # 若是不为空
    #     print("Using {} predictors".format(n_pred))  # 选择使用多少个预测基因
    #     potential_pred = VMR.sort_values(ascending=False).index[:n_pred]  # 将VMR从大到小排列进行排列 进行从0进行选取到n_pred
    covariance_matrix = pd.DataFrame(np.abs(np.corrcoef(raw.T.loc[potential_pred])),index=potential_pred,columns=potential_pred).fillna(0)  # 将空值设置为0    # 通过raw.T将行列转置， 通过.loc[list[]]截取列表中指定的行号     通过np.corrcoef(行为基因列,列为细胞列) 计算Peason相关系数,  np.abs()对每个值取绝对值  np.corrcoef()计算的是行基因之间的相关系数矩阵
    return covariance_matrix  # 返回协方差矩阵

def setTarget(data):
    n_subsets=int(data.shape[1]/300)
    target_gene=np.random.choice(data.columns,[n_subsets,300],replace=False)
    return target_gene

def setPredict(Target_gene,covar_matrix,ntop=5):
    predicts=[]
    for i,target in enumerate(Target_gene):
        gene_not_in_target=np.setdiff1d(covar_matrix.columns,target)
        # print(target,covar_matrix)
        subMatrix=(covar_matrix.loc[target,gene_not_in_target])

        sorted_idx=np.argsort(-subMatrix.values,axis=1)

        predict=subMatrix.columns[sorted_idx[:,:ntop].flatten()]

        predicts.append(predict)
        print('{} ：predict_gene,{} ： target_gene'.format(len(predict),len(target)))
    return predicts

def DataSet_process(data):
    # inspect_data(data)
    data=pd.DataFrame(data) #----------------------------------------------------------------- 注意输入
    gene_metric = (data.var() / (1 + data.mean())).sort_values(ascending=False)  # dataframe.var()表示求每列的样本方差(比方差除的时候少一个样本),dataframe.mena()表示求每列的样本均值,降序排列   计算方差和平均比率,选择丢弃是否
    # gene_metric = gene_metric[gene_metric > 0]  # 只挑选列值>0的行  此时gene_metric是一个Series  而且删除后series前面的行号不变
    gene_to_impute=filter_genes(gene_metric,0.5)
    data=data.loc[:,gene_to_impute]
    print('data: ', data)
    data.columns=[i for i in range(len(gene_to_impute))]
    covariance_matrix = get_distance_matrix(data)

    Target_gene=setTarget(data)
    Predict_gene=setPredict(Target_gene,covariance_matrix)
    # print(Target_gene,Predict_gene)

    return Target_gene,Predict_gene,gene_to_impute,data


# if __name__=='__main__':
#     print(int(5/2))
#     data=None
#     DataSet_process(data)