import math
import os
import time
import tracemalloc
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
import autoencoder
import autoencoder_clustering_trainable_clustering
import ACT_Network
from utils_edit_PCA import clustering
import check as ch
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import numpy as np


def cosine_similarity(arr1, arr2):


    dot_product = np.dot(arr1, arr2)
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)

    cosine_sim = dot_product / (norm_arr1 * norm_arr2)

    return cosine_sim




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'

n_clusters = 7
#typess='Sample ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,,Marker score,Marker score\n'
#typess='Sample ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'
database='DLPFC'
files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
files=[ '151507','151508', '151509', '151510', '151669', '151670', '151671', '151672','151673', '151674', '151675', '151676']
files=['151673']
#files=[ '151507']
#typess=' - ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
typess=' - ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'

for i in files:
    tracemalloc.start()
    start_time=time.time()



    dataset = 'E:\Project_Large_Datasets\ST\\'+database+'\\'+i

    file_fold =  str(dataset) #please replace 'file_fold' with the download path
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    print(adata
    )
    if 'highly_variable' not in adata.var.keys():
        simple_preprocess.preprocess(adata)

    if 'feat' not in adata.obsm.keys():
        simple_preprocess.get_feature(adata)

    image = cv2.imread('E:\\Project_Large_Datasets\\ST\\DLPFC\\'+i+'\\'+i+'_full_image.tif')


    points = adata.obsm['spatial']
    position = adata.obsm['spatial']
    d={}
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
        d[i5] = ngs


    tile_size = 120  # e.g., 20x20 pixels

    tiles = []
    model = models.resnet50(pretrained=True)
    model.eval()
    iujk=0
    embe=[]
    # for point in points:
    #
    #     x, y = point
    #     print(point)
    #     tile = image[y - tile_size // 2:y + tile_size // 2, x - tile_size // 2:x + tile_size // 2]
    #     #img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    #     preprocess = transforms.Compose([
    #
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     iujk=iujk+1
    #
    #     img_tensor = preprocess(tile).unsqueeze(0)
    #     # if iujk==4:
    #     #     break
    #     # Generate embeddings
    #     with torch.no_grad():
    #         embeddings = model(img_tensor)
    #         embeddings=embeddings.numpy().tolist()
    #         embe.append(embeddings[0])


    ijp=0

    pl=[]
    length1=len(points)
    for i5 in range(length1):
        p= adata.obsm['feat'][i5]

        smv=[]
        for i7 in d[i5]:
            g = adata.obsm['feat'][i7]
            smv.append(adata.obsm['feat'][i7])
            #smv.append(cosine_similarity(p,g))

        p=np.append(p,np.asarray(smv))
        #p=np.asarray(smv)
        #pl.append(p)





    #npg = np.asarray(pl, dtype=np.float32)
    #np.save(i+'.npy', npa)
    npa = np.load(i+'.npy')

    #_________
    #adata.obsm['feat'] = npg
    #_________
    #adata.obsm['feat'] = np.append(adata.obsm['feat'], npg, axis=1)
    #_________
    #adata.obsm['feat'] = np.append(adata.obsm['feat'], points / 13000, axis=1)
    #_________
    #adata.obsm['feat'] = np.append(adata.obsm['feat'], npa, axis=1)
    #_________
    #adata.obsm['feat'] = np.append(npa, points/13000, axis=1)
    #_________
    # adata.obsm['feat'] = np.append(adata.obsm['feat'], points / 13000, axis=1)
    # adata.obsm['feat'] = np.append(adata.obsm['feat'], npa, axis=1)


    adata.obsm['emb']=  adata.obsm['feat']
    df_meta = pd.read_csv(file_fold + '/metadata_check.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    print(df_meta_layer.values)
    ijl=[]
    for ijkl in df_meta_layer.values:
        p=[1, 0, 0, 0, 0, 0, 0]
        if ijkl=='WM':
            p = [1, 0, 0, 0, 0, 0, 0]
        if ijkl=='Layer1':
            p = [0, 1, 0, 0, 0, 0, 0]
        if ijkl=='Layer2':
            p = [0, 0, 1, 0, 0, 0, 0]
        if ijkl=='Layer3':
            p = [0, 0, 0, 1, 0, 0, 0]
        if ijkl=='Layer4':
            p = [0, 0, 0, 0, 1, 0, 0]
        if ijkl=='Layer5':
            p = [0, 0, 0, 0, 0, 1, 0]
        if ijkl=='Layer6':
            p = [0, 0, 0, 0, 0, 0, 1]
        ijl.append(p)

    ijl = np.asarray(ijl, dtype=np.float32)

    adata.obs['ground_truth'] = df_meta_layer.values
    #adata.obs['ground_truth_label'] = ijl
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    print(adata.obs['ground_truth'].shape)
    print(adata.obs['ground_truth'])
    radius = 50

    tool = 'mclust'

    model = autoencoder_clustering_trainable_clustering.GraphST(adata=adata, device=device)

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



        ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
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

    # sc.pl.spatial(adata,
    #               img_key="hires",
    #               color=["ground_truth", "domain"],
    #               title=["Ground truth", "ARI=%.4f"%ARI],
    #               show=True)

print(typess)