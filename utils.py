import os
import torch
import umap
import numpy as np
import pandas as pd
import scanpy as sc, anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

def split_input(input_data, source_name, target_name):
    """
    :param input_data:
    :return: source_data, source_cellinfo, target_data
    """
    source_data = input_data[input_data.obs['batch'] == source_name, :]
    target_data = input_data[input_data.obs['batch'] == target_name, :]
    return source_data, target_data

def matrix_one_hot(x, class_count):
    # torch.eye() Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    return torch.eye(class_count)[x,:]

def pre_proccess(adata, target_sum=1e4):
    sc.pp.normalize_total(adata,target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata.X, adata

def cluster_metrics(data:sc.AnnData):
    """
    :param data:
    :return: nmi, ari
    """
    # ARI NMI metrics
    data_x = data.X
    cellinfo = data.obs['CellType']
    fit = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, metric='cosine', random_state=123)
    output_code = fit.fit_transform(data_x)
    class_num = len(np.unique(cellinfo))
    kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=666)
    y_pred = kmeans.fit_predict(output_code)
    y_true = cellinfo

    nmi_k = nmi_score(y_true, y_pred)
    ari_k = ari_score(y_true, y_pred)
    return nmi_k, ari_k

def csv2h5ad(inPath, outPath):
    batch_fa = 0.6
    de_facScale = 0.4
    dropout_mid = int(1) # changed
    seed = int(100) #100

    target_expr = pd.read_csv(inPath + 'counts_batch' + str(batch_fa) + '_mid' + str(dropout_mid) +
                                  "_facScale" +str(de_facScale) + "_seed" + str(seed)+ '_b2.csv', index_col=0)
    target_cellinfo = pd.read_csv(inPath + 'label_batch' + str(batch_fa) + '_mid' + str(dropout_mid) +
                              "_facScale" +str(de_facScale) + "_seed" + str(seed)+ '_b2.csv')
    target_cellinfo = target_cellinfo['x'].values
    target_data = sc.AnnData(target_expr)
    target_data.obs['CellType'] = target_cellinfo.astype(str)
    target_data.obs['batch'] = 'batch1'

    # source simulated data
    source_expr = pd.read_csv(inPath + 'counts_batch' + str(batch_fa) + '_mid' + str(dropout_mid) +
                              "_facScale" +str(de_facScale) + "_seed" + str(seed)+ '_b1.csv')
    source_cellinfo = pd.read_csv(inPath + 'label_batch' + str(batch_fa) + '_mid' + str(dropout_mid) +
                              "_facScale" +str(de_facScale) + "_seed" + str(seed)+ '_b1.csv')
    source_cellinfo = source_cellinfo['x'].values
    source_data = sc.AnnData(source_expr)
    source_data.obs['CellType'] = source_cellinfo.astype(str)
    source_data.obs['batch'] = 'batch2'

    # concat and wirte to h5ad
    simulated_drop1 = ad.concat([source_data, target_data])
    simulated_drop1.write_h5ad(os.path.join(outPath, 'simulated_drop1.h5ad'))
    print('csv2h5ad done!')
    

