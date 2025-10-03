import os
import traceback
from turtle import shape

import anndata
from sklearn import metrics

import check as ch
import  scanpy as sc


dataset = 'E:\Project_Large_Datasets\ST\DLPFC\\151507'

method='GraphST'
domain_key="domain"
dataset='E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC\\151507.h5ad'

file_fold = str(dataset) #please replace 'file_fold' with the download path

adata=anndata.read_h5ad(file_fold)

print(adata)
print(adata.X)

print("<<<<<<45>>>>>>>>>")

method='conST'
domain_key="domain"
files=os.listdir('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC')

moran=''
geary=''
filess=''
files=['151507.h5ad']
for i in files:
    adata=anndata.read_h5ad('E:\Project_Large_Datasets\ST_Results\\'+method+'\\DLPFC\\'+i)
    print(adata)
    print(adata.X)
    print(adata.obs)
    # print(adata.var)
    # print(adata.uns)
    # print(adata.obsm)
    # print(adata.X)

    # print(adata)
    # chaos = ch.compute_CHAOS(adata, 'refine')
    # print(chaos)
    # pas = ch.compute_PAS(adata, 'refine')
    # print(pas)
    # aws = ch.compute_ASW(adata, 'refine')
    # print(aws)







#
    try:
       moranI, gearyC = ch.marker_score(adata, domain_key)
    except:
        traceback.print_exc()
        moranI, gearyC = 'TBD' , 'TBD'
    print(moranI)
#
#
#     moran=moran+','+str(moranI)
#     geary=geary+','+str(gearyC)
#     filess=filess+","+str(i)
#     print(moran)
#     print(filess)
#
#
# print(filess)
# print(moran)
# print(geary)
