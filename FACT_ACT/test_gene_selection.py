import os
import time
import tracemalloc
import torch
import autoencoder_model
import scanpy as sc
import pandas as pd
from sklearn import metrics
import simple_preprocess_gene_selection
import autoencoder_gene_selection
from utils_gene_selection import clustering
import check as ch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'

n_clusters = 7
#typess='Sample ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,,Marker score,Marker score\n'
#typess='Sample ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'
database='DLPFC'
files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
#files=[ '151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672','151673', '151674', '151675', '151676']
files=['151673']
#typess=' - ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
typess=' - ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'

for i in files:
    tracemalloc.start()
    start_time=time.time()


    dataset = 'E:\Project_Large_Datasets\ST\\'+database+'\\'+i

    file_fold =  str(dataset) #please replace 'file_fold' with the download path
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    if 'highly_variable' not in adata.var.keys():
        simple_preprocess_gene_selection.preprocess(adata)

    if 'feat' not in adata.obsm.keys():
        simple_preprocess_gene_selection.get_feature(adata)
    #adata.obsm['feat']= adata.obsm['feat'][:,0:1500]
    adata.obsm['emb'] = adata.obsm['feat']
    print(adata.obsm['emb'].shape)
    radius = 50

    tool = 'mclust'

    model = autoencoder_gene_selection.GraphST(adata, device=device)

    # train model
    adata = model.train()


    # clustering
    if tool == 'mclust':
       values=clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.

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
        df_meta_layer = df_meta['layer_guess']
        adata.obs['ground_truth'] = df_meta_layer.values

        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
        nmi=metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
        hom=metrics.homogeneity_score(adata.obs['domain'], adata.obs['ground_truth'])
        cm=metrics.completeness_score(adata.obs['domain'], adata.obs['ground_truth'])
        moranI, gearyC = ch.marker_score(adata, 'domain')
        chaos = ch.compute_CHAOS(adata, 'domain')
        pas = ch.compute_PAS(adata, 'domain')
        aws = ch.compute_ASW(adata, 'domain')


        typess=typess+i+','+str(ARI)+','+str(nmi)+','+str(hom)+','+str(cm)+','+str(chaos)+','+str(pas)+','+str(aws)+','+str(during_time)+','+str(memory)+','+str(moranI)+','+str(gearyC)+'\n'
    else:
        typess = typess + i + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str(
            "TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str("TBD") + ',' + str(
            "TBD") + ',' + str("TBD") + '\n'
    print("ARI:::")
    print(ARI)

    # sc.pl.spatial(adata,
    #               img_key="hires",
    #               color=["ground_truth", "domain"],
    #               title=["Ground truth", "ARI=%.4f"%ARI],
    #               show=True)

print(typess)