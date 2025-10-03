import os
import time
import tracemalloc
from torchvision import models, transforms
import numpy as np
import torch
import cv2
from PIL import Image
import autoencoder_model
import scanpy as sc
import pandas as pd
from sklearn import metrics
import simple_preprocess
import autoencoder
import autoencoder_clustering_trainable_clustering
from utils_edit_PCA import clustering
import check as ch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.2'

n_clusters = 7
#typess='Sample ,Accuracy,Accuracy,Accuracy,Continuity,Continuity,Continuity,,Marker score,Marker score\n'
#typess='Sample ,ARI,nmi,hom,cm,chaos,pas,aws,time,space,moranI,gearyC\n'
database='DLPFC'
files=os.listdir('E:\Project_Large_Datasets\ST\\'+database)
files=[ '151507','151508', '151509', '151510', '151669', '151670', '151671', '151672','151673', '151674', '151675', '151676']
#files=['151673']
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

    # Define points (replace with your actual points)
    points = adata.obsm['spatial']
    # points=points.tolist()
    # print(points)


    # Define the size of the tile
    tile_size = 120  # e.g., 20x20 pixels

    tiles = []
    model = models.resnet50(pretrained=True)
    model.eval()
    iujk=0
    embe=[]
    for point in points:

        x, y = point
        print(point)
        tile = image[y - tile_size // 2:y + tile_size // 2, x - tile_size // 2:x + tile_size // 2]
        #img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        preprocess = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        iujk=iujk+1

        img_tensor = preprocess(tile).unsqueeze(0)
        # if iujk==4:
        #     break
        # Generate embeddings
        with torch.no_grad():
            embeddings = model(img_tensor)
            embeddings=embeddings.numpy().tolist()
            embe.append(embeddings[0])


    npa = np.asarray(embe, dtype=np.float32)
    np.save(i+'.npy', npa)



print(typess)