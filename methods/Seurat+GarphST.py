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

# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'
# the number of clusters
n_clusters = 7


database='DLPFC'
files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
files=['151671']
typess=' - ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
typess=' - ,ARI,nmi,hom,cm,chaos,pas,aws,moranI,gearyC\n'

for i in files:
    tracemalloc.start()
    start_time=time.time()
    dataset = 'E:\Project_Large_Datasets\ST\\'+database+'\\'+i
    # read data
    file_fold =  str(dataset) #please replace 'file_fold' with the download path
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    print('<<<<<<1')
    print(adata)
    adata.var_names_make_unique()
    # define model
    print('<<<<<<2')
    print(adata)
    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # read the annotation
    print('<<<<<<3')
    print(adata)
    Ann_df = pd.read_csv(os.path.join('E:\\Project_Large_Datasets\\ST\\DLPFC\\', i, i+'_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    print(adata.obs_names)
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    model = GraphST.GraphST(adata, device=device)

    # train model
    adata = model.train()
    # set radius to specify the number of neighbors considered during refinement
    radius = 50
    print('<<<<<<4')
    print(adata)
    tool = 'mclust' # mclust, leiden, and louvain

    # clustering


    if tool == 'mclust':
       clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
       clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

    print('<<<<<<5')
    print(adata)

    # add ground_truth
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    # filter out NA nodes

    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print('<<<<<<6')
    print(adata)


    adata.write_h5ad(
        'E:\Project_Large_Datasets\ST_Results\GraphST\\'+database+'\\'+i+'.h5ad',
    )
    # calculate metric ARI


    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    print(ARI)
    nmi=metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
    print(nmi)
    hom=metrics.homogeneity_score(adata.obs['domain'], adata.obs['ground_truth'])
    print(hom)
    cm=metrics.completeness_score(adata.obs['domain'], adata.obs['ground_truth'])
    print(cm)
    chaos=ch.compute_CHAOS(adata,'domain')
    print(chaos)
    pas = ch.compute_PAS(adata, 'domain')
    print(pas)
    aws = ch.compute_ASW(adata, 'domain')
    print(aws)

    moranI, gearyC = ch.marker_score(adata, 'domain')
    print('Dataset:', dataset)
    print('ARI:', ARI)
    print(nmi)
    print(hom)
    print(cm)
    print(chaos)
    print(moranI)
    print(gearyC)

    typess=typess+i+','+str(ARI)+','+str(nmi)+','+str(hom)+','+str(cm)+','+str(chaos)+','+str(pas)+','+str(aws)+','+str(moranI)+','+str(gearyC)+'\n'
    #print(typess)
    # adata.uns['ARI'] = ARI
    # # plotting spatial clustering result
    # sc.pl.spatial(adata,
    #               img_key="hires",
    #               color=["ground_truth", "domain"],
    #               title=["Ground truth", "ARI=%.4f"%ARI],
    #               show=True)
    # output_df = pd.DataFrame([[nmi,hom,cm,chaos,pas,aws,moranI,gearyC]],
    #                index = ['GraphST'],#your method name
    #                columns=[['Accuracy','Accuracy','Accuracy','Continuity','Continuity','Continuity','Marker score','Marker score'],
    #                         ['NMI','HOM','COM','CHAOS','PAS','ASW','Moran\'I','Geary\'s C']])
    # output_df.to_excel('output_result.xlsx')
print(typess)
