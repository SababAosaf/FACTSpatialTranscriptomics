import os
import traceback
from turtle import shape

import anndata
from sklearn import metrics

import check as ch
import  scanpy as sc
from scanpy import read_10x_h5
import pandas as pd

method='DeepST'
files=os.listdir('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC')

typess=' Sample ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
typess=' Sample ,ARI,nmi,hom,cm,chaos,pas,aws,moranI,gearyC\n'

database="DLPFC"
for i in files:

    main_file=anndata.read_h5ad('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC\\'+i)

    chaos = ch.compute_CHAOS(main_file, 'DeepST_refine_domain')
    pas = ch.compute_PAS(main_file, 'DeepST_refine_domain')
    aws = ch.compute_ASW(main_file, 'DeepST_refine_domain')

    try:
       moranI, gearyC = ch.marker_score(main_file, "DeepST_refine_domain")
    except Exception as e:
        traceback.print_exc()
        moranI, gearyC = 'TBD' , 'TBD'


    # ARI = metrics.adjusted_rand_score(main_file.obs['DeepST_refine_domain'], main_file.obs['ground_truth'])
    # nmi=metrics.normalized_mutual_info_score(main_file.obs['DeepST_refine_domain'], main_file.obs['ground_truth'])
    # hom=metrics.homogeneity_score(main_file.obs['DeepST_refine_domain'], main_file.obs['ground_truth'])
    # cm=metrics.completeness_score(main_file.obs['DeepST_refine_domain'], main_file.obs['ground_truth'])

    # typess = typess + i[0:i.index('.')] + ',' + str(ARI) + ',' + str(nmi) + ',' + str(hom) + ',' + str(cm) + ',' + str(
    #     chaos) + ',' + str(
    #     pas) + ',' + str(aws) + ',' + str(moranI) + ',' + str(gearyC) + '\n'

    typess = typess + i[0:i.index('.')] + ',' + str(
        chaos) + ',' + str(
        pas) + ',' + str(aws) + ',' + str(moranI) + ',' + str(gearyC) + '\n'


print(typess)



