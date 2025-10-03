import sys
import ot

from sklearn.cluster import  DBSCAN
from sklearn.datasets import make_blobs
import torch
from numpy import set_printoptions
#import check as ch
from preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from autoencoder_model_cluster import Encoder, Encoder_sparse, Encoder_map, Encoder_sc
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from fast_pytorch_kmeans import KMeans
from utils_edit_PCA import clustering, mclust_training


class GraphST():
    def __init__(self, 
        adata,
        neigh,
        distance_matrix,
        labels=None ,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=3000,
        dim_input=10000,
        dim_output=32,   ####
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X'
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning. 
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning. 
            The default is 1.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional    
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data. 
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata.copy()

        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        self.neigh=neigh
        fix_seed(self.random_seed)


        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)


        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)

        print(self.adata.obsm['feat'])
        # print("HERE : 1.1")
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)



        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        # self.primary_labels = torch.FloatTensor(self.adata.obs['domain']).to(self.device)
        self.adj = self.adata.obsm['adj']
        # self.spatialx_ = torch.FloatTensor(self.adata.obsm['spatialx']).to(self.device)
        # self.spatialy_ = torch.FloatTensor(self.adata.obsm['spatialy']).to(self.device)
        # self.spatialx=self.adata.obsm['spatialx']
        # self.spatialy = self.adata.obsm['spatialy']
        # np.savetxt('n.txt', self.adata.obsm['graph_neigh'],fmt='%i')
        # np.savetxt('aj.txt', self.adata.obsm['adj'],fmt='%i')

        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        # print("HERE : 1.2")
        self.distance_matrix=torch.from_numpy(distance_matrix)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           # print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           # print("HERE : 1.21")
           self.adj = preprocess_adj(self.adj)
           # print("HERE : 1.22")
           self.adj = torch.FloatTensor(self.adj).to(self.device)


        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs

    # print("HERE : 1.3")


    def train(self):

        if self.datatype in ['Stereo', 'Slide']:
           self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output ,self.graph_neigh).to(self.device)
            self.loss_CSL = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('Begin to train ST data...')


        self.model.train()

        clusr=0

        # Convert data to PyTorch tensor
        data, _ = make_blobs(n_samples=self.features.shape[0], centers=7, cluster_std=0.60, random_state=0)

        # Convert data to PyTorch tensor
        tensor_data = torch.from_numpy(data).float()
        centroids = tensor_data[torch.randperm(tensor_data.size(0))[:7]]
        #UPDATED
        self.epochs=3000
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            self.features_a = permutation(self.features)
            self.hiden_feat, self.decoded,self.labels = self.model(self.features, self.features_a, self.adj)
            #self.hiden_feat, self.decoded = self.model(self.features, self.features_a, self.adj)



            #if  False:
            if   epoch >= 1000 and epoch<2000:
                if epoch ==1000:
                    self.model.weight1.requires_grad = False
                if epoch%100==0:
                    self.adata.obsm['emb'] = self.hiden_feat.detach().cpu().numpy()
                    radius = 50
                    tool = 'mclust'

                    # Types by CLustering on new data
                    values = clustering(self.adata, 7, radius=radius, method=tool)

                    old_type = self.adata.obs['domain'].values

                    for i in range(len(self.adata.obs['domain'])):
                        tipes=[]
                        # TAKE TYPE OF NEIGHBOURS
                        for j in self.neigh[i]:
                            tipes.append(old_type[j])

                        if tipes.count(tipes[0]) == len(tipes) or tipes.count(tipes[0])>4:
                            if old_type[i] != tipes[0]:

                                self.adata.obs['domain'][i] = tipes[0]

                    current_labels = F.one_hot(torch.FloatTensor(self.adata.obs['domain']).long()-1,7).float().to(self.device)


                loss = F.mse_loss(self.labels,current_labels)
                print("CLUST: "+str(loss))

                # n_neigh = 50
                # new_type = []
                # old_type = self.adata.obs['domain'].values
                #
                #
                #
                # # calculate distance
                # old_features = self.hiden_feat.clone().detach()
                # new_features=self.hiden_feat.clone().detach()
                # n_cell = self.distance_matrix.shape[0]
                #
                # count=0
                # for i in range(n_cell):
                #     tipes=[]
                #     for j in self.neigh[i]:
                #         tipes.append(old_type[j])
                #     if tipes.count(tipes[0]) == len(tipes):
                #         if old_type[i] != tipes[0]:
                #             pass
                #
                #
                #     # vec = self.distance_matrix[i, :]
                #     # index = vec.argsort()
                #     # neigh_type = []
                #     #
                #     # for j in range(1, n_neigh + 1):
                #     #     neigh_type.append(old_type[index[j]])
                #     #
                #     # max_type = max(neigh_type, key=neigh_type.count)
                #     # new_type.append(max_type)
                #
                #     # if max_type!= old_type[i]:
                #     #
                #     #     count+=1
                #     #     for j in range(1, n_neigh + 1):
                #     #         if old_type[index[j]]==max_type:
                #     #             new_features[i] = old_features[index[j]]
                #
                # print("FEATURE: " + str(count))
                # l2=F.mse_loss(self.hiden_feat, new_features)

                #loss = self.alpha * self.loss_feat+self.alpha*l2
            #elif False :
            elif epoch>=2000:
                if epoch % 50==000:
                    self.model.weight1.requires_grad = True
                    self.adata.obsm['emb'] = self.hiden_feat.detach().cpu().numpy()
                    radius = 50
                    tool = 'mclust'
                    values = clustering(self.adata, 7, radius=radius, method=tool)
                    old_type = self.adata.obs['domain'].values

                    for i in range(len(self.adata.obs['domain'])):
                        tipes=[]
                        for j in self.neigh[i]:
                            tipes.append(old_type[j])
                        if tipes.count(tipes[0]) == len(tipes) or tipes.count(tipes[0])>4:
                            if old_type[i] != tipes[0]:
                                self.adata.obs['domain'][i] = tipes[0]

                    current_labels = F.one_hot(torch.FloatTensor(self.adata.obs['domain']).long()-1,7).float().to(self.device)

                #loss1 = F.mse_loss(self.labels, current_labels)
                #print(self.labels)
                #print(current_labels)

                loss1= F.kl_div(self.labels.log(), current_labels, reduction='batchmean')
                self.loss_feat = F.mse_loss(self.features, self.decoded)

                loss=self.alpha * self.loss_feat+self.alpha *loss1 * 2

                print("CLUST: "+str(loss))
            else:
                self.loss_feat = F.mse_loss(self.features, self.decoded)
                loss = self.alpha * self.loss_feat


            # n_neigh = 50
            # new_type = []
            # old_type = values
            #
            #
            # n_cell = self.distance_matrix.shape[0]
            # for i in range(n_cell):
            #     vec = self.distance_matrix[i, :]
            #     index = vec.argsort()
            #     neigh_type = []
            #     for j in range(1, n_neigh + 1):
            #         neigh_type.append(old_type[index[j]])
            #     max_type = max(neigh_type, key=neigh_type.count)
            #     new_type.append(max_type)
            #     print(max_type)


            #new_type = [str(i) for i in list(new_type)]

            # self.sp1= F.mse_loss(self.spatialtestx,self.spatialx_)
            # self.sp2 = F.mse_loss(self.spatialtesty, self.spatialy_)
            #loss = self.alpha * self.loss_feat+self.alpha * self.loss_labels

            # if epoch>2950:
            #     kmeans = KMeans(n_clusters=7)
            #     labels = kmeans.fit_predict(self.features)
            #     num_classes = labels.max().item() + 1
            #     one_hot_tensor = torch.nn.functional.one_hot(labels, num_classes=num_classes)
            #     one_hot_tensor=one_hot_tensor.type(torch.cuda.FloatTensor)
            #     self.loss_labels = F.mse_loss(self.labels, one_hot_tensor)
            #     loss = self.loss_labels* self.alpha +self.alpha * self.loss_feat
            # else:


            #
            # print(self.loss_feat)
            # print(self.sp1)
            # print(self.sp2)

            #loss =  self.alpha * self.loss_labels
            ###############      MetaHeuristic TRY
            # if 1==0 and epoch > 200 and epoch % 500 == 0:
            #
            #     self.emb_rec = self.model(self.features, self.features_a, self.adj)[0].detach().cpu().numpy()
            #     self.adata.obsm['emb'] = self.emb_rec
            #     ij=clustering(self.adata, 7, radius=50, method='mclust', refinement=True)#,start=0.1, end=0.5, increment=0.01)
            #     if ij=="CORRECT":
            #         clusr1 = ch.compute_ASW( self.adata, 'domain')
            #     else:
            #         clusr1=100
            #     self.model.weight1_s=self.model.weight1
            #     with torch.no_grad():
            #         print(self.model.weight1.size())
            #         totals=30000*32
            #         parcent=3.125
            #         for i in range(0,int(totals*parcent/100)):
            #             a=random.randrange(0,30000)
            #             b=random.randrange(0,32)
            #             p=random.choice([0,1])
            #             if p==0:
            #                 self.model.weight1[a][b]=self.model.weight1[a][b]+self.model.weight1[a][b]*0.1
            #             else:
            #                self.model.weight1[a][b] = self.model.weight1[a][b] - self.model.weight1[a][b] * 0.1
            #     self.emb_rec = self.model(self.features, self.features_a, self.adj)[0].detach().cpu().numpy()
            #     self.adata.obsm['emb'] = self.emb_rec
            #     clustering(self.adata, 7, radius=50, method='mclust', refinement=True)#,start=0.1, end=0.5, increment=0.01)
            #     if ij == "CORRECT":
            #         clusr2 = ch.compute_ASW(self.adata, 'domain')
            #     else:
            #         clusr2 = 100
            #     if clusr1<=clusr2:
            #         self.model.weight1= self.model.weight1_s

            ###############      KMEANS TRY
            # if  epoch % 200 == 0:
            #     kmeans = KMeans(n_clusters=7)
            #
            #     features_cpu = self.labels.detach().cpu().numpy()
            #     #features_np = features_cpu.numpy()
            #     cluster_labels = kmeans.fit_predict(features_cpu)
            #     cluster_labels = torch.tensor(cluster_labels, device=self.hiden_feat.device)
            #
            #     loss_fn = SilhouetteLoss()
            #     loss_1 = loss_fn(self.labels, cluster_labels)
            #     loss = loss + self.alpha * loss_1
            #     print("FEATU")


            ###############      DBScan
            # if  1==2 and epoch % 100 == 0:
            #     dbscan = DBSCAN(eps=0.5, min_samples=500)
            #     features_cpu = self.hiden_feat.detach().cpu().numpy()
            #     #features_np = features_cpu.numpy()
            #     print(features_cpu)
            #     cluster_labels = dbscan.fit_predict(features_cpu)
            #     cluster_labels = torch.tensor(cluster_labels, device=self.hiden_feat.device)
            #
            #     loss_fn = SilhouetteLoss()
            #     loss_1 = loss_fn(self.hiden_feat, cluster_labels)
            #     loss = loss + self.alpha * loss_1
            #     print("FEATU")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print("Optimization finished for ST data!")

        # self.emb_rec = self.model(self.features, self.features_a, self.adj)[0].detach().cpu().numpy()
        # self.adata.obsm['emb'] = torch.Tensor.cpu(self.features).numpy()
        self.emb_rec = self.model(self.features, self.features_a, self.adj)[0].detach().cpu().numpy()
        self.adata.obsm['emb'] = self.emb_rec
        print("ADATA")
        print(self.adata)
        print("EMB")
        numpy_array=self.model.weight1.to('cpu')
        numpy_array = numpy_array.detach().numpy()
        df = pd.DataFrame(numpy_array)
        df.to_csv('tensor_data.csv', index=False)


        print(self.adata.obsm['emb'])

        return self.adata
         
    def train_sc(self):
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        
        print('Begin to train scRNA data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            
            emb = self.model_sc(self.feat_sc)
            loss = F.mse_loss(emb, self.feat_sc)
            
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
            
        print("Optimization finished for cell representation learning!")
        
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
         
            return emb_sc
        
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        
        # Normalize features for consistence between ST and scRNA-seq
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
          
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()

            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
             
            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE 

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
            
        print("Mapping matrix learning finished!")
        
        # take final softmax w/o computing gradients
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell
            
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell

            return self.adata, self.adata_sc
    
    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M


class SilhouetteLoss(nn.Module):
    def __init__(self):
        super(SilhouetteLoss, self).__init__()

    def forward(self, features, labels):
        """
        Args:
            features (torch.Tensor): The data points (N x D) where N is the number of samples and D is the dimension.
            labels (torch.Tensor): The cluster labels for each point (N x 1).

        Returns:
            torch.Tensor: The silhouette-based loss value.
        """
        N = features.size(0)
        intra_cluster_distances = torch.zeros(N)
        nearest_cluster_distances = torch.zeros(N)

        for i in range(N):
            # Current point and its cluster label
            point = features[i]
            label = labels[i]

            # Intra-cluster distance (distance to other points in the same cluster)
            same_cluster = (labels == label).nonzero().squeeze(1)
            if len(same_cluster) > 1:
                intra_distances = torch.norm(features[same_cluster] - point, dim=1)
                intra_cluster_distances[i] = torch.mean(intra_distances)

            # Nearest-cluster distance (distance to the closest point in a different cluster)
            different_cluster = (labels != label).nonzero().squeeze(1)
            if len(different_cluster) > 0:
                nearest_distances = torch.norm(features[different_cluster] - point, dim=1)
                nearest_cluster_distances[i] = torch.min(nearest_distances)

        # Compute silhouette score: (b - a) / max(a, b)
        a = intra_cluster_distances
        b = nearest_cluster_distances
        silhouette_scores = (b - a) / torch.max(a, b)

        # Return a loss that minimizes negative silhouette scores
        loss = -torch.mean(silhouette_scores)
        return loss


class ClustLoss(nn.Module):
    def __init__(self):
        super(ClustLoss, self).__init__()

    def forward(self, features,points):
        """
        Args:
            features (torch.Tensor): The data points (N x D) where N is the number of samples and D is the dimension.
            labels (torch.Tensor): The cluster labels for each point (N x 1).

        Returns:
            torch.Tensor: The silhouette-based loss value.
        """

        distance_matrix = ot.dist(features, features, metric='euclidean')
        # Create a sample tensor

        # Normalize the tensor to the range [0, 1]
        tensor_min = distance_matrix.min()
        tensor_max = distance_matrix.max()
        normalized_distance_matrix = (distance_matrix - tensor_min) / (tensor_max - tensor_min)

        # Flatten the tensors to 1D
        tensor1_flat = points.view(-1)
        tensor2_flat = normalized_distance_matrix.view(-1)



        return F.mse_loss(tensor1_flat,tensor2_flat)