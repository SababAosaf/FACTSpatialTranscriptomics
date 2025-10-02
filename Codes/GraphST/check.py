import scanpy as sc
import pandas as pd
import squidpy as sq
import numpy as np
from scipy.spatial import *
from sklearn.preprocessing import *

from sklearn.metrics import *
from scipy.spatial.distance import *


def marker_score(adata, domain_key, top_n=5):

    adata = adata.copy()

    count_dict = adata.obs[domain_key].value_counts()

    adata = adata[adata.obs[domain_key].isin(count_dict.keys()[count_dict > 3].values)]

    sc.pp.normalize_per_cell(adata)

    sc.pp.log1p(adata)

    sc.tl.rank_genes_groups(adata, groupby=domain_key)

    selected_genes = []
    for i in range(top_n):
        toadd = list(adata.uns['rank_genes_groups']['names'][i])
        selected_genes.extend(toadd)

    selected_genes = np.unique(selected_genes)
    sq.gr.spatial_neighbors(adata)

    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        genes=selected_genes,
        n_perms=100,
        n_jobs=1,
        show_progress_bar=False,
    )

    moranI = np.median(adata.uns["moranI"]['I'])

    sq.gr.spatial_autocorr(
        adata,
        mode="geary",
        genes=selected_genes,
        n_perms=100,
        n_jobs=1,
        show_progress_bar=False,
    )

    moranI = np.median(adata.uns["moranI"]['I'])
    gearyC = np.median(adata.uns["gearyC"]['C'])
    return moranI, gearyC


def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0


def _compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results) / len(clusterlabel)


def markerFC(adata_valid, marker_list, sdm_key):
    rst_dict = {}
    sdm_unique = adata_valid.obs[sdm_key].cat.categories
    for marker in marker_list:
        mean_exp_list = []
        for sdm in sdm_unique:
            mean_exp_list.append(np.mean(adata_valid[adata_valid.obs[sdm_key] == sdm][:, marker].X))
        max_sdm_idx = np.argmax(mean_exp_list)
        #         print(sdm_unique[max_sdm_idx])

        max_sdm_value = np.max(mean_exp_list)
        other_sdm_value = np.mean(adata_valid[adata_valid.obs[sdm_key] != sdm_unique[max_sdm_idx]][:, marker].X)
        cur_fc = max_sdm_value / other_sdm_value
        rst_dict[marker] = cur_fc
    return rst_dict

def compute_CHAOS(adata,pred_key,spatial_key='spatial'):
    return _compute_CHAOS(adata.obs[pred_key],adata.obsm[spatial_key])

def compute_PAS(adata,pred_key,spatial_key='spatial'):
    return _compute_PAS(adata.obs[pred_key],adata.obsm[spatial_key])

def compute_ASW(adata,pred_key,spatial_key='spatial'):
    d = squareform(pdist(adata.obsm[spatial_key]))
    return silhouette_score(X=d,labels=adata.obs[pred_key],metric='precomputed')
