import os
import time
import tracemalloc

import torch
import scanpy as sc
import GraphST
import pandas as pd
from sklearn import metrics
from utils import clustering
import check as ch
# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'
n_clusters = 7
database='DLPFC'
i='151507'



dataset = 'E:\Project_Large_Datasets\ST\\'+database+'\\'+i

file_fold =  str(dataset)
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

print(adata)

model = GraphST.GraphST(adata, device=device)
print("THIS IS ")
print(model.adata.X)
adata = model.train()
print(adata)


#
clustering(adata, n_clusters, radius=50, method='mclust', refinement=True)
print(adata)
#
#
# # add ground_truth
# df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
# df_meta_layer = df_meta['layer_guess']
# adata.obs['ground_truth'] = df_meta_layer.values
#
#
# # filter out NA nodes
# adata = adata[~pd.isnull(adata.obs['ground_truth'])]
# print(adata.obs['domain'])



