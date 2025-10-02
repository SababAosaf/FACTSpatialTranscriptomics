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
# print(torch.cuda.is_available())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'
# the number of clusters
n_clusters = 7


# database='DLPFC'
# files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
# files=['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
# files=['151673']
# ground_truth='layer_guess'

database='moredata'
files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
ground_truth='ground_truth'

typess=' - ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
typess=' - ,ARI,nmi,hom,cm,chaos,pas,aws,moranI,gearyC\n'
typess=' - ,time,space\n'


for i in files:

    tracemalloc.start()
    start_time=time.time()

    dataset = 'E:\Project_Large_Datasets\ST\\'+database+'\\'+i
    if database=='moredata':
        file_fold = str(dataset)
        df_metas = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
        unique_values_count = df_metas['ground_truth'].nunique()
        n_clusters = unique_values_count
        if n_clusters>21:n_clusters=12


    # read data
    file_fold =  str(dataset) #please replace 'file_fold' with the download path
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # define model

    model = GraphST.GraphST(adata, device=device)

    # train model
    adata = model.train()
    # set radius to specify the number of neighbors considered during refinement
    radius = 50

    tool = 'mclust' # mclust, leiden, and louvain

    # clustering
    if tool == 'mclust':
       clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
       clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta[ground_truth]
    adata.obs['ground_truth'] = df_meta_layer.values
    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    # print(adata.obs['domain'])

    end_time = time.time()
    during = end_time - start_time
    size, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory = peak / 1024 / 1024
    during_time = during
    # adata.write_h5ad(
    #     'E:\Project_Large_Datasets\ST_Results\GraphST\\'+database+'\\'+i+'.h5ad',
    # )
    # calculate metric ARI


    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    print("ARI: "+str(ARI))
    typess=typess+i+','+str(during_time)+','+str(memory)+"\n"
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
    #
    # moranI, gearyC = ch.marker_score(adata, 'domain')
    # print('Dataset:', dataset)
    # print('ARI:', ARI)
    # print(nmi)
    # print(hom)
    # print(cm)
    # print(chaos)
    # print(moranI)
    # print(gearyC)
    # print("CKECK: 1")
    typess=typess+i+','+str(ARI)+','+str(nmi)+','+str(hom)+','+str(cm)+','+str(chaos)+','+str(pas)+','+str(aws)+','+str(during)+','+str(memory)+'\n'
    # #print(typess)
    # adata.uns['ARI'] = ARI
    # # plotting spatial clustering result



    custom_palette = [
        '#e176c1', '#8c564b', '#9467bd', '#d62728',
        '#2ca02c', '#fb7e12', '#1f77b4',
        '#17becf', '#ff7f0e', '#ffbb78', '#98df8a', '#2b83ba', '#bcbd22', '#d62728', '#7f7f7f', '#b084cc',
        '#56b4e9', '#cc79a7', '#f0e442', '#c6c8e4'
    ]

    sc.pl.spatial(adata,palette=custom_palette,
                  img_key="hires",
                  color=["ground_truth", "domain"],
                  title=["Ground truth", "ARI=%.4f"%ARI],
                  show=True)
    # output_df = pd.DataFrame([[nmi,hom,cm,chaos,pas,aws,moranI,gearyC]],
    #                index = ['GraphST'],#your method name
    #                columns=[['Accuracy','Accuracy','Accuracy','Continuity','Continuity','Continuity','Marker score','Marker score'],
    #                         ['NMI','HOM','COM','CHAOS','PAS','ASW','Moran\'I','Geary\'s C']])
    # output_df.to_excel('output_result.xlsx')
print(typess)
