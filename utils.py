import scanpy as sc
import torch
import umap
import numpy as np
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
