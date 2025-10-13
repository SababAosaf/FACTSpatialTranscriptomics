
import math
import os
import pickle
import time
import tracemalloc
import random
from collections import Counter
import numpy
from torchvision import models, transforms
import numpy as np
import torch
import cv2
import ot
from PIL import Image
import autoencoder_model
import scanpy as sc
import pandas as pd
from sklearn import metrics
import simple_preprocess


import ACT_Network
print(55)
from utils_edit_PCA import clustering

import check as ch
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def cosine_similarity(arr1, arr2):


    dot_product = np.dot(arr1, arr2)
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)

    cosine_sim = dot_product / (norm_arr1 * norm_arr2)

    return cosine_sim


print(55)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'
n_clusters = 7

#typess='Sample ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,,Marker score,Marker score\n'
#typess='Sample ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'


# database='DLPFC'
# files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
# files=[ '151507','151508', '151509', '151510', '151669', '151670', '151671', '151672','151673', '151674', '151675', '151676']
# files=['151673']
# ground_truth='layer_guess'


database='moredata'
files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
ground_truth='ground_truth'

#typess=' - ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
c_ = ' - ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'
typess= c_

for i in files:



    tracemalloc.start()
    start_time=time.time()
    radius = 50
    tool = 'mclust'
    #                       READ
    dataset = 'E:\Project_Large_Datasets\ST\\'+database+'\\'+i

    if database=='moredata':
        file_fold = str(dataset)
        df_metas = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
        unique_values_count = df_metas['ground_truth'].nunique()
        n_clusters = unique_values_count
        if n_clusters>21:n_clusters=12


    file_fold =  str(dataset) #please replace 'file_fold' with the download path
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    points = adata.obsm['spatial']
    position = adata.obsm['spatial']


    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=30000)
    adata_Vars = adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    distance_matrix = ot.dist(position, position, metric='euclidean')

    d_big={}
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])

    for i5 in range(n_spot):
        vec = distance_matrix[i5, :]
        distance = vec.argsort()
        ngs=[]

        for t in range(1, 6 + 1):

            y = distance[t]
            ngs.append(y)

        d_big[i5] = ngs
    d_small = {}


    for i5 in d_big.keys():
        li=[]
        for ij in d_big[i5]:
            first=i5
            neigh=ij
            li.append((neigh,distance_matrix[i5][ij]))
        sorted_list = sorted(li, key=lambda x: x[1])
        first_d=sorted_list[0][1]
        l2=[]
        for ijk in sorted_list:
            if ijk[1]<first_d*1.5:
                l2.append(ijk[0])
        d_small[i5] = l2



    if 'highly_variable' not in adata.var.keys():
        simple_preprocess.preprocess(adata)
    if 'feat' not in adata.obsm.keys():
        simple_preprocess.get_feature(adata)


    adata.obsm['emb']=  adata.obsm['feat']

    print("GRAPHST")
    model = autoencoder_clustering_feeding.GraphST(adata,d_small, distance_matrix,device=device)

    # train model
    adata = model.train()


    # clustering
    #adata.obsm['emb'] = np.append(adata.obsm['emb'],  points/13000, axis=1)

    if tool == 'mclust':
       #values=clustering(adata, 7, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
       values = clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
    elif tool in ['leiden', 'louvain']:
       clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)



    if values == "CORRECT":
        end_time = time.time()
        during = end_time - start_time
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory = peak / 1024 / 1024
        during_time = during

        df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
        df_meta_layer = df_meta[ground_truth]
        adata.obs['ground_truth'] = df_meta_layer.values

        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])

        # sc.pl.spatial(adata,
        #               img_key="hires",
        #               color=["ground_truth", "UNREFINED" ,"domain"],
        #               title=["Ground truth", "??" ,"ARI=%.4f" % ARI],
        #               show=True,save=True)

        nmi=metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
        hom=metrics.homogeneity_score(adata.obs['domain'], adata.obs['ground_truth'])
        cm=metrics.completeness_score(adata.obs['domain'], adata.obs['ground_truth'])
        moranI, gearyC = ch.marker_score(adata, 'domain')
        chaos = ch.compute_CHAOS(adata, 'domain')
        pas = ch.compute_PAS(adata, 'domain')
        aws = ch.compute_ASW(adata, 'domain')
        print(ARI)

        typess=typess+i+','+str(ARI)+','+str(nmi)+','+str(hom)+','+str(cm)+','+str(chaos)+','+str(pas)+','+str(aws)+','+str(during_time)+','+str(memory)+','+str(moranI)+','+str(gearyC)+'\n'

    else:
        typess = typess + i + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str(
            "TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str(
            "TBD") + ',' + str("TBD") + '\n'
    print("ARI:::")
    print(ARI)

    sc.pl.spatial(adata,
                  img_key="hires",
                  color=["ground_truth", "domain"],
                  title=["Ground truth", "ARI=%.4f"%ARI],
                  show=True)

print(typess)