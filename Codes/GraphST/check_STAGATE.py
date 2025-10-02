import os
import traceback
from turtle import shape

import anndata
from sklearn import metrics

import check as ch
import  scanpy as sc
from scanpy import read_10x_h5
import pandas as pd

method='STAGATE'
files=os.listdir('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC')
typess=' Sample ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
typess=' Sample ,ARI,nmi,hom,cm,chaos,pas,aws,moranI,gearyC\n'
database="DLPFC"

for i in files:

    main_file = sc.read_visium('E:\Project_Large_Datasets\ST\\DLPFC\\'+i[0:i.index('.')], count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata=anndata.read_h5ad('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC\\'+i)
    # print('E:\Project_Large_Datasets\ST\\DLPFC\\'+i[0:i.index('.')])
    # print('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC\\'+i)
    # print(adata.obs)
    domains=[]

    array_row=[]
    array_column=[]

    for ij in main_file.obs['array_row']:
        array_row.append(ij)
    for ij in main_file.obs['array_col']:
        array_column.append(ij)



    array_row_adata = []
    array_column_adata = []
    mclust=[]

    for ij in adata.obs['array_row']:
        array_row_adata.append(ij)
    for ij in adata.obs['array_col']:
        array_column_adata.append(ij)
    for ij in adata.obs['mclust']:
        mclust.append(ij)

    for ij in range(0,len(array_row)):

        x=array_row[ij]
        y=array_column[ij]

        this_item = '1'
        for ik in range(0,len(array_row_adata)):
            x1 = array_row_adata[ik]
            y1 = array_column_adata[ik]
            if x==x1 and y==y1:
                this_item=mclust[ik]
        domains.append(str(this_item))


    # for ij in range(len(array_row_adata)):
    #     print((array_column[ij],array_row[ij],domains[ij]))
    #
    # for ij in range(len(array_row)):
    #     print((array_column[ij],array_row[ij],domains[ij]))



    main_file.obs['domain'] = domains
    main_file.var_names_make_unique()

    # print(main_file.obs)
    # print(adata.obs)

    chaos = ch.compute_CHAOS(main_file, 'domain')
    pas = ch.compute_PAS(main_file, 'domain')
    aws = ch.compute_ASW(main_file, 'domain')

    try:
       moranI, gearyC = ch.marker_score(main_file, "domain")
    except Exception as e:
        traceback.print_exc()
        moranI, gearyC = 'TBD' , 'TBD'

    # df_meta = pd.read_csv('E:\Project_Large_Datasets\ST\DLPFC\\'+ i[0:i.index('.')]+ '/metadata.tsv', sep='\t')
    # df_meta_layer = df_meta['layer_guess']
    #
    # main_file.obs['ground_truth'] = df_meta_layer.values
    # main_file = main_file[~pd.isnull(main_file.obs['ground_truth'])]

    # ARI = metrics.adjusted_rand_score(main_file.obs['domain'], main_file.obs['ground_truth'])
    # sc.pl.spatial(main_file,
    #               img_key="hires",
    #               color=["ground_truth", "domain"],
    #               title=["Ground truth", "ARI=%.4f" % ARI],
    #               show=True)
    # nmi=metrics.normalized_mutual_info_score(main_file.obs['domain'], main_file.obs['ground_truth'])
    # hom=metrics.homogeneity_score(main_file.obs['domain'], main_file.obs['ground_truth'])
    # cm=metrics.completeness_score(main_file.obs['domain'], main_file.obs['ground_truth'])

    # typess = typess + i[0:i.index('.')] + ',' + str(ARI) + ',' + str(nmi) + ',' + str(hom) + ',' + str(cm) + ',' + str(
    #     chaos) + ',' + str(
    #     pas) + ',' + str(aws) + ',' + str(moranI) + ',' + str(gearyC) + '\n'

    # typess = typess + i+","+ str(moranI) + ',' + str(gearyC) + '\n'
    typess = typess + i+","+ str(chaos) + ',' + str(pas)  + ','+ str(aws) + '\n'
print(typess)


