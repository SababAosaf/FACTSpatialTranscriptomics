
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
import methods.autoencoder_model
import scanpy as sc
import pandas as pd
from sklearn import metrics
import methods.simple_preprocess


import methods.ACT_Network

from methods.utils_edit_PCA import clustering

import methods.check as ch
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


def ACT(database):

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


    files=os.listdir(database)
    ground_truth = 'layer_guess'

    if database=='DLPFC':
        ground_truth = 'layer_guess'
    else:
        ground_truth='ground_truth'

    #typess=' - ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,Marker score,Marker score\n'
    c_ = ' - ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'
    typess= c_

    files = ['151673']
    for i in files:
        tracemalloc.start()
        start_time=time.time()
        radius = 50
        tool = 'mclust'
        #                       READ
        dataset = database+'\\'+i
        file_fold =  str(dataset) #please replace 'file_fold' with the download path
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        if database!='DLPFC':
            file_fold = str(dataset)
            df_metas = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
            unique_values_count = df_metas['ground_truth'].nunique()
            n_clusters = unique_values_count
            if n_clusters>21:n_clusters=12
        points = adata.obsm['spatial']
        position = adata.obsm['spatial']
        #__________________________________________SPATIAL COORDINATES INCLUSION__________________________________________
        #
        #
        # x=points[:,0]
        # y=points[:,1]
        # min_value_x = np.min(x)
        # min_value_y = np.min(y)
        # x = (x - min_value_x)// 5
        # y = (y - min_value_y)// 5
        # max_value_x = np.max(x)
        # max_value_y = np.max(y)
        # array = np.zeros((len(points),max_value_x+1),dtype=float)
        # array1 = np.zeros((len(points),max_value_y+1),dtype=float)
        #
        #
        # for i55 in range(len(points)):
        #     array[i55,x[i55]] = 1
        #     for jk in range (1,5):
        #         if x[i55]-jk>-1:
        #             array[i55, x[i55]-jk]=1-jk*0.25
        #         if x[i55]+jk<max_value_x+1:
        #             array[i55, x[i55]+jk]=1-jk*0.25
        #     array1[i55, y[i55]] = 1
        #     for jk in range (1,5):
        #         if y[i55]-jk>-1:
        #             array1[i55, y[i55]-jk]=1-jk*0.25
        #         if y[i55]+jk<max_value_y+1:
        #             array1[i55, y[i55]+jk]=1-jk*0.25
        #
        #
        # adata.obsm['spatialx']=array
        # adata.obsm['spatialy'] = array1

        # unique_values = np.unique(x)
        # num_unique_values = len(unique_values)
        # ///////////////////////////__________________SPATIAL COORDINATES INCLUSION__________________________________________


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



        #______________________________________MANUAL ANNOTATION__________________________________________________

        # df_meta = pd.read_csv('E:\\Project_Large_Datasets\\ST\\DLPFC\\151673' + '/metadata_check.tsv', sep='\t')
        # df_meta_layer = df_meta['layer_guess']
        # ijl=[]
        # for ijkl in df_meta_layer.values:
        #     p=[1, 0, 0, 0, 0, 0, 0]
        #     if ijkl=='WM':
        #         p = [1, 0, 0, 0, 0, 0, 0]
        #     if ijkl=='Layer1':
        #         p = [0, 1, 0, 0, 0, 0, 0]
        #     if ijkl=='Layer2':
        #         p = [0, 0, 1, 0, 0, 0, 0]
        #     if ijkl=='Layer3':
        #         p = [0, 0, 0, 1, 0, 0, 0]
        #     if ijkl=='Layer4':
        #         p = [0, 0, 0, 0, 1, 0, 0]
        #     if ijkl=='Layer5':
        #         p = [0, 0, 0, 0, 0, 1, 0]
        #     if ijkl=='Layer6':
        #         p = [0, 0, 0, 0, 0, 0, 1]
        #     ijl.append(p)
        # ijl = np.asarray(ijl, dtype=np.float32)
        # adata.obs['ground_truth'] = df_meta_layer.values
        # adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        #<////////////////>__________________MANUAL ANNOTATION__________________________________________________


        #_____________________________ Scatter method____________________________________
        # for pk in range(1,500):
        #     low = 0  # lower bound
        #     high = len(feat)-1  # upper bound
        #     num_random_numbers = 200  # number of random numbers
        #     random_numbers = [random.randint(low, high) for _ in range(num_random_numbers)]
        #     id=[]
        #     for ikj in random_numbers:
        #         ip=feat[ikj]
        #         sorted_indices = np.argsort(ip)[::-1]
        #         top_n = 50
        #         highest_indices = sorted_indices[:top_n]
        #         id.extend(highest_indices.tolist())
        #
        #     counter = Counter(id)
        #     most_common_number = counter.most_common(10)
        #
        #
        #
        #     for value, count in most_common_number:
        #         adata.var['highly_variable'][value] = False

        # <////////////////>_________________ Scatter method____________________________________


        # for pk in range(1,500):
        #     low = 0  # lower bound
        #     high = len(feat)-1  # upper bound
        #     num_random_numbers = 200  # number of random numbers
        #     random_number = random.randint(low, high)
        #     forf= d_big[random_number]
        #     forf.append(random_number)
        #     id=[]
        #     c = feat[forf[0]]
        #     p=1
        #     for ikj in forf:
        #         if p==1:
        #             p=2
        #             continue
        #         ip=feat[ikj]
        #         c=numpy.add(c,ip)
        #     counter = Counter(id)
        #     most_common_number=counter.most_common(10)
        #     for value, count in most_common_number:
        #         adata.var['highly_variable'][value] = False




        #### ijkk=0
        #### dict12={
        ####     'WM' :[],
        ####     'Layer1': [],
        ####     'Layer2': [],
        ####     'Layer3': [],'Layer4' :[],
        ####     'Layer5': [],
        ####     'Layer6': [],
        #### }

        #### for i in feat:
        ####     dict12[adata.obs['ground_truth'][ijkk]].append(i)
        ####     ijkk=ijkk+1



        #### c=0
        #### for ijk in dict12.keys():
        ####     arrays=dict12[ijk]
        ####     c=arrays[0]
        ####     p=0
        ####     for ijkh in arrays:
        ####         if p==0:
        ####             p=1
        ####             continue
        ####         c=np.add(c,ijkh)



        ####     sorted_indices = np.argsort(c)[::-1]
        ####     top_n = 10
        ####     highest_indices = sorted_indices[:top_n]
        ####     print(highest_indices)



        if 'highly_variable' not in adata.var.keys():
            methods.simple_preprocess.preprocess(adata)
        if 'feat' not in adata.obsm.keys():
            methods.simple_preprocess.get_feature(adata)


        # adata.obsm['emb'] = adata.obsm['feat']
        # clustering(adata, 7, radius=radius, method=tool, refinement=True)
        # features = adata.obs['domain']

        # ________________________________________ NEIGHBOURS________________________________

        # d={}
        # for i5 in range(n_spot):
        #     vec = distance_matrix[i5, :]
        #     distance = vec.argsort()
        #     ngs=[]
        #     for t in range(1, 6 + 1):
        #         y = distance[t]
        #         ngs.append(y)
        #     d[i5] = ngs

        #<///////////////>________________________ NEIGHBOURS________________________________


        # ________________________________________ IMAGE EMBEEDING________________________________

        # image = cv2.imread('E:\\Project_Large_Datasets\\ST\\DLPFC\\'+i+'\\'+i+'_full_image.tif')
        # tile_size = 120  # e.g., 20x20 pixels
        # tiles = []
        # model = models.resnet50(pretrained=True)
        # model.eval()
        # iujk=0
        # embe=[]
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

        # </////////////>___________________________ IMAGE EMBEEDING________________________________


        #___________________________________________ NEIGHBOUR COSINE SIMILARITY________________________________

        # ijp=0
        #
        # pl=[]
        # length1=len(points)
        # for i5 in range(length1):
        #     p= adata.obsm['feat'][i5]
        #
        #     smv=[]
        #     for i7 in d[i5]:
        #         g = adata.obsm['feat'][i7]
        #         smv.append(adata.obsm['feat'][i7])
        #         #smv.append(cosine_similarity(p,g))
        #
        #     p=np.append(p,np.asarray(smv))
        #     pl.append(p)

        #</////////////>__________________________ NEIGHBOUR COSINE SIMILARITY________________________________



        # npg = np.asarray(pl, dtype=np.float32)
        # #np.save(i+'.npy', npa)
        # npa = np.load(i+'.npy')


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
        model = methods.ACT_Network.Embedding_Network(adata,d_small, distance_matrix,device=device)

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