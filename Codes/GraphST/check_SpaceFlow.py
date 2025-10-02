import os
import traceback
from turtle import shape

import anndata
from sklearn import metrics

import check as ch
import  scanpy as sc
from scanpy import read_10x_h5
import pandas as pd

method='SpaceFlow'
files=os.listdir('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC')
typess=' Sample ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
typess=' Sample ,ARI,nmi,hom,cm,chaos,pas,aws,moranI,gearyC\n'
database="DLPFC"

for i in files:
    if '.txt' not in i:
        continue
    main_file = sc.read_visium('E:\Project_Large_Datasets\ST\\DLPFC\\'+i[0:i.index('.')], count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata=open( 'E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC\\'+i, "r").read()

    adata=adata.split("\n")
    adata=adata[0:len(adata)-1]
    main_file.obs['domain']=adata
    main_file.var_names_make_unique()

    chaos = ch.compute_CHAOS(main_file, 'domain')
    pas = ch.compute_PAS(main_file, 'domain')
    aws = ch.compute_ASW(main_file, 'domain')
    print(main_file.obs['domain'])
    try:
       moranI, gearyC = ch.marker_score(main_file, "domain")
    except Exception as e:
        traceback.print_exc()
        moranI, gearyC = 'TBD' , 'TBD'

    df_meta = pd.read_csv('E:\Project_Large_Datasets\ST\DLPFC\\'+ i[0:i.index('.')]+ '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']

    main_file.obs['ground_truth'] = df_meta_layer.values
    main_file = main_file[~pd.isnull(main_file.obs['ground_truth'])]

    # ARI = metrics.adjusted_rand_score(main_file.obs['domain'], main_file.obs['ground_truth'])
    # nmi=metrics.normalized_mutual_info_score(main_file.obs['domain'], main_file.obs['ground_truth'])
    # hom=metrics.homogeneity_score(main_file.obs['domain'], main_file.obs['ground_truth'])
    # cm=metrics.completeness_score(main_file.obs['domain'], main_file.obs['ground_truth'])

    # typess = typess + i[0:i.index('.')] + ',' + str(ARI) + ',' + str(nmi) + ',' + str(hom) + ',' + str(cm) + ',' + str(
    #     chaos) + ',' + str(
    #     pas) + ',' + str(aws) + ',' + str(moranI) + ',' + str(gearyC) + '\n'

    typess = typess + i[0:i.index('.')] + ',' + str(
        chaos) + ',' + str(
        pas) + ',' + str(aws) + ',' + str(moranI) + ',' + str(gearyC) + '\n'

print(typess)


