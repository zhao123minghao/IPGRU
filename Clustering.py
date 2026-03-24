#复杂数据集

import scimpute
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering,KMeans
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import manifold, datasets
from scipy import ndimage
from Data_Process import read_datas
from sklearn.manifold import TSNE
# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import pandas as pd
from sklearn import metrics

def visualize_all_genes(X, Y, G, p):
    ''' generate plots using all genes
    Parameters
        ------------
                X: input data matrix; genes in columns (same below)
                Y: imputed data matrix
                G: ground truth
                p: parameters

        Return
        -----------
                None
        '''

    # histograms of gene expression
    max_expression = max(G.values.max(), X.values.max(), Y.values.max())
    min_expression = min(G.values.min(), X.values.min(), Y.values.min())
    print('\n max expression:', max_expression)
    print('\n min expression:', min_expression)

    scimpute.hist_df(
        Y, xlab='Expression', title='Imputation({})'.format(p.name_imputation),
        dir=p.tag, range=[min_expression, max_expression])
    scimpute.hist_df(
        X, xlab='Expression', title='Input({})'.format(p.name_input),
        dir=p.tag, range=[min_expression, max_expression])
    scimpute.hist_df(
        G, xlab='Expression', title='Ground Truth({})'.format(p.name_ground_truth),
        dir=p.tag, range=[min_expression, max_expression])

    # histograms of correlations between genes in imputation and ground truth
    # and of correlations between cells in imputation and ground truth
    # when ground truth is not provide,
    # input is used as ground truth
    print('\n> Correlations between ground truth and imputation')
    print('ground truth dimension: ', G.shape, 'imputation dimension: ', Y.shape)
    print('generating histogram for correlations of genes between ground truth and imputation')
    scimpute.hist_2matrix_corr(
        G.values, Y.values,
        title="Correlation for each gene\n(Ground_truth vs Imputation)\n{}\n{}".
            format(p.name_ground_truth, p.name_imputation),
        dir=p.tag, mode='column-wise', nz_mode='first'  # or ignore
    )

    print('generating histogram for correlations of cells between ground truth and imputation')
    scimpute.hist_2matrix_corr(
        G.values, Y.values,
        title="Correlation for each cell\n(Ground_truth vs Imputation)\n{}\n{}".
            format(p.name_ground_truth, p.name_imputation),
        dir=p.tag, mode='row-wise', nz_mode='first'
    )

    #  heatmaps of data matrices
    print('\n> Generating heatmaps of data matrices')
    range_max, range_min = scimpute.max_min_element_in_arrs([Y.values, G.values, X.values])
    print('\nrange:', range_max, ' ', range_min)

    scimpute.heatmap_vis(Y.values,
                         title='Imputation ({})'.format(p.name_imputation),
                         xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

    scimpute.heatmap_vis(X.values,
                         title='Input ({})'.format(p.name_input),
                         xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

    scimpute.heatmap_vis(G.values,
                         title='Ground_truth ({})'.format(p.name_ground_truth),
                         xlab='Genes', ylab='Cells', vmax=range_max, vmin=range_min, dir=p.tag)

    # PCA and tSNE plots
    print('\n> Generating PCA and tSNE plots')
    if p.cluster_file is not None:
        cluster_info = scimpute.read_data_into_cell_row(p.cluster_file)
    # cluster_info = cluster_info.astype('str')
    else:
        cluster_info = None

    scimpute.pca_tsne(df_cell_row=Y, cluster_info=cluster_info,
                      title=p.name_imputation, dir=p.tag)
    scimpute.pca_tsne(df_cell_row=X, cluster_info=cluster_info,
                      title=p.name_input, dir=p.tag)
    scimpute.pca_tsne(df_cell_row=G, cluster_info=cluster_info,
                      title=p.name_ground_truth, dir=p.tag)

#k_value_selection
def K_value_selection(data):
    d=[]
    linkages = ['ward', 'average', 'complete']
    for i in range(1,100):
        km=AgglomerativeClustering(linkage=linkages[2],n_clusters=i)
        km.fit(data)
        d.append(km.inertia_)
    plt.plot(range(1,100),d,marker='o')
    plt.xlabel('number of clusters')
    plt.ylabel('SSE')
    plt.show()
    return max(d)
#层次聚类是通过可视化然后人为去判断大致聚为几类，很明显在共同父节点的一颗子树可以被聚类为一个类

def plot_scatter(X,  color, alpha=0.5):
    return plt.scatter(X[:, 0],X[:, 1],c=color,alpha=alpha,edgecolor='k')

def clustering(raw):
    info=pd.read_csv('/home/yangtianyu/ljs_data/Single_Cell_Data/Dataset/PBMC_G949_10K/cell_type_labels/clusters.csv')
    # scimpute.pca_tsne(df_cell_row=raw, cluster_info=info,
    #                   title='cluter', dir='home/ljs/GAAN')
    # data=1.0*(raw-raw.mean())/raw.std()     #犯错了，包标准化的数据再次归一化
    data=np.array(raw)
    data1=np.array(raw)
    print('data: \n',data)
    linkages=['ward','average','complete']
    # n_cluter=K_value_selection(data)
    # agg=AgglomerativeClustering(linkage=linkages[0],n_clusters=10)
    agg=KMeans(n_clusters=10,random_state=0)
    agg.fit(data)
    pre_labels=agg.labels_
    print(data.shape)
    print(pre_labels.shape)
    index = [i for i in range(data.shape[0])]
    data=pd.DataFrame(data=data,index=index)
    labels=pd.DataFrame(data=pre_labels,index=index,columns=['聚类类别'])
    print('data: \n',data)
    print('is nan: \n',np.isnan(data).any())
    tsne=TSNE()
    tsne.fit_transform(data)        #tsne相当于是对多维数据进行降维到二维，然后将多个样本投影到一张坐标系图上，适合再聚类完后进行数据可视化
    tsne=pd.DataFrame(tsne.embedding_,index=data.index)
    #绘图

    plt.figure(figsize=(12, 12))  #错误了，先创建了一个图，结果又用这条语句创建了一张空白图
    # print('tsne: ',tsne[0:3])
    print('tsne_size : ',tsne.index.size)
    print('labels: ',labels)
    d = tsne[labels[u'聚类类别'] == 0]      #返回布尔类型的index的规模的数组，当聚类类别为0时 该行为True  否则为False
    print(labels[u'聚类类别']==0)
    print(d[0],d[1])
    plt.plot(d[0], d[1],'o', color='r')
    d = tsne[labels[u'聚类类别'] == 1]
    plt.plot(d[0], d[1], 'o',color='g')
    d = tsne[labels[u'聚类类别'] == 2]
    plt.plot(d[0], d[1], 'o',color='b')
    d = tsne[labels[u'聚类类别'] == 3]
    plt.plot(d[0], d[1], 'o',color='c')
    d = tsne[labels[u'聚类类别'] == 4]
    plt.plot(d[0], d[1], 'o',color='m')
    d = tsne[labels[u'聚类类别'] == 5]
    plt.plot(d[0], d[1],'o' ,color='y')
    d = tsne[labels[u'聚类类别'] == 6]
    plt.plot(d[0], d[1], 'o',color='k')
    d = tsne[labels[u'聚类类别'] == 7]
    plt.plot(d[0], d[1], 'o',color='navy')
    d = tsne[labels[u'聚类类别'] == 8]
    plt.plot(d[0], d[1], 'o',color='darkcyan')
    d = tsne[labels[u'聚类类别'] == 9]
    plt.plot(d[0], d[1], 'o',color='chocolate')
    # plt.subplot(111)
    # data=np.array(data)
    # plot_scatter(data,labels)
    plt.title('Magic')
    plt.show()
    plt.savefig('Magic.jpg')
    raw=pd.DataFrame(data=raw,index=index)
    raw=raw[labels[u'聚类类别']==2]
    "/home/yangtianyu/ljs_data/Single_Cell_Data/Dataset/PBMC_G949_10K/cell_type_labels/"
    #读取数据
    # label=pd.read_csv('/mnt/qwe2/ljs_data/Single_Cell_Data/Dataset/PBMC_G949_10K/cell_type_labels/clusters.csv')
    # label=label.iloc[:,]
    # print('labels: ',labels,'label: ',label)
    # labels=np.array(labels)
    # label=np.array(label)
    # real_label=search()
    # print('pre_labels: ',pre_labels,'real_label: ',real_label)
    score=metrics.calinski_harabasz_score(data1,pre_labels)
    print('score: ',score)

    return raw

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y

def read_data():
    X, y = datasets.load_digits(return_X_y=True)
    n_samples, n_features = X.shape
    X,y=nudge_images(X,y)
    print(X,y)
    return X

if __name__=='__main__':
    input_matrix, gene_ids, cell_ids =read_datas()
    m, n = input_matrix.shape  # input_matrix 是sparse矩阵
    # input_matrix=input_matrix.T
    raw=input_matrix.todense()

    # data=input_matrix.todense()
    clustering(raw)