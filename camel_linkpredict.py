#author: alex sun
#date: 01/18/2021
#desc: link prediction on camels
#this script does the following
#collect static attributes from CAMELS dataset
#train an AE on static attributes
#use the AE encoder to project to latent space
#calculate node distance
#formulate network by applying percentile cutoff
#generate adjacency matrix and save it as sparse matrix
#plot the graph
#apply kmeans to generate clustering
#02052021: add metric choice, modified normaliztion method
#===========================================================================
import os.path as osp

import argparse
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import pickle as pkl

import torch_geometric.transforms as T
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
import sys
from sklearn.cluster import KMeans
from readcamels import getStaticAttr,getSubSet4Clustering,getSubSet
import networkx as netx
import scipy
import hydrostats as Hydrostats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(2021119)
random.seed(2021119)
torch.manual_seed(20188)

class MyDataset(Dataset):
    """Subclass of PyTorch's Dataset
    """
    def __init__(self, data, transform=None):
        self.data_size = data.shape[0]
        self.input = data
        self.transform = transform

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.input[idx,:]
        sample = torch.tensor(sample,dtype=torch.float)
        if self.transform:
            sample = self.transform(sample) 
        sample = F.normalize(sample, p=1, dim=-1)
        return sample

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.encoder=nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),    
            nn.Linear(16, latent_dim),
            nn.ReLU(True),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(True),    
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),     
        )

    def forward(self, features):
        x = self.encoder(features)
        x = self.decoder(x)

        return x

def add_noise(x):
    noise = torch.randn(x.size()) * 1e-6
    noisy_img = x + noise
    return noisy_img

def genAdjacency(featureMat, perc=98,smeasure='euclidean'):
    """Generate pairwise distances
    1/20: note pytorchgeometric can import distance directly
    Parameters
    ----------
    featureMat: matrix of features in columsn
    perc: threshold for cutoff
    smeasure: similarity measure to use for defining the adjacency 
    """
    from scipy.spatial.distance import cdist
    import sklearn.preprocessing as skp

    assert(smeasure in ["euclidean","mahalanobis","jensenshannon", "correlation"])
    D = cdist(featureMat, featureMat,smeasure)

    np.fill_diagonal(D,1)
    D = np.reciprocal(D) #get inverse distance
    #should we do this?
    row_normalize=False
    if row_normalize:
        #row normalization
        np.fill_diagonal(D,0) #exclude diagonal in the sum 
        sum_of_rows = D.sum(axis=1)
        D = D / sum_of_rows[:, np.newaxis]
        np.fill_diagonal(D,1.0)    
    else:
        #do max/min     
        np.fill_diagonal(D,0) #exclude diagonal
        amin,amax = (np.min(D),np.max(D)   )  
        D = (D-amin)/(amax-amin)        
        np.fill_diagonal(D,1.0)    
        
    #calculate histogram
    Dup=np.triu(D)
    pnt = np.percentile(Dup,perc)
    print (f'{perc} percentile is {pnt}')
    #these are distance so smaller ones are closer
    D[D<pnt] = 0.0
    A = np.copy(D)
    A[A>0.0] = 1.0
    id = np.where(A==1.0)[0]
    print ('percent connected {:.2f}'.format(len(id)/(A.shape[0]*A.shape[0])*100.0))

    return A,D

def train(df, reTrain=False, latent_dim  = 4):
    """Train autoencoder
    Parameters
    ---------
    latent_dim, dimension of the latent variable
    """
    dataset = MyDataset(df.to_numpy(),transform=None)
    if reTrain:
        dataloader = DataLoader(dataset, batch_size=32, 
                        shuffle=True,drop_last=False,num_workers=4)

    testdataloader = DataLoader(dataset, batch_size=1, 
                        shuffle=False,drop_last=False,num_workers=4)

    model = AE(input_dim=df.shape[1], latent_dim=latent_dim)
    model.to(device)
    model_path = f'models/camels_ae_{latent_dim}.pth'

    if reTrain:
        #train
        lr = 0.005
        nEpoch=100
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()


        model.train()    
        for i in range(nEpoch):
            loss=0.0
            for x in dataloader:
                x = x.view(x.size(0), -1)    
        
                optimizer.zero_grad()
                x= x.to(device)
                outputs = model(x)
                train_loss = criterion(x,outputs)
                train_loss.backward()

                loss+=train_loss.item()
                optimizer.step()
            loss = loss/len(dataloader)
            print (f'Epoch {i}, loss={loss:.4f}')
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    latentMat = np.zeros((df.shape[0],latent_dim))
    for idx,x in enumerate(testdataloader):
        x = x.view(x.size(0), -1)  
        x=x.to(device)
        out = model.encoder(x)

        latentMat[idx,:] = out.data.cpu().numpy().squeeze()
    return latentMat

def plotStations(allDF,cvec,n_clusters):
    allDF['clusterid'] = cvec
    base = plt.cm.get_cmap('cubehelix')
    color_list = base(np.linspace(0, 1, n_clusters))
    cmap_name = base.name + str(n_clusters)
    cmap = base.from_list(cmap_name, color_list, n_clusters)    
    plt.figure(figsize=(12,8))
    x = allDF['lng']
    y = allDF['lat']
    print (x)
    plt.scatter(x,y,c=cvec,cmap=cmap)
    plt.savefig('camels_cluster.png')
    plt.close()

def drawGraph(G, pos, nodelist=None,nodeval=None,allDF=None,prefix='camels_graph'):
    from mpl_toolkits.basemap import Basemap as Basemap
    m = Basemap(
            epsg=4326,
            llcrnrlon=-130,
            llcrnrlat=25,
            urcrnrlon=-60,
            urcrnrlat=50,
            lat_ts=0,
            resolution='i',
            suppress_ticks=False)

    if nodelist is None:
        netx.draw(G, pos)
    else:
        Blues = plt.get_cmap('RdBu')
        for inode in nodelist:
            fig,ax=plt.subplots(1,1,figsize=(8,6))
            alist = [n for n in G.neighbors(inode)]
            if nodeval is None:
                nodecolors=['blue' for n in G.neighbors(inode)]
            else:
                nodecolors=[Blues(nodeval[a_neighbor]) for a_neighbor in G.neighbors(inode)]
            alist.append(inode)
            if nodeval is None:
                nodecolors.append('red')
            else:
                nodecolors.append(Blues(nodeval[inode]))

            elist = [e for e in G.edges(inode)]
            netx.draw(G, pos, ax=ax, nodelist=alist, node_color=nodecolors, 
                    edgelist=elist,node_size=50, vmin=-1.0, vmax=1.0)
            m.readshapefile('maps/conus', 'conus',ax=ax,color='k')
            m.readshapefile('maps/statesp020', 'states',ax=ax,color='gray')
            if not allDF is None:
                gageid = allDF.loc[allDF.index[inode], ['gauge_str']].to_string(header=False,index=False)
                plt.title('gage {0}'.format(gageid))
    
            plt.savefig('{1}_{0}.png'.format(inode,prefix))
            plt.close()

def genGraph(A,allDF):
    G=netx.from_numpy_matrix(A)
    x = allDF['lng'].to_numpy()
    y = allDF['lat'].to_numpy()
    n_node = A.shape[0]
    pos={anode:(x[anode],y[anode]) for anode in range(n_node)}
    return G,pos

def plotDegreeMap(A,**kwargs):
    cutoff = kwargs['cutoff']
    latenttype=kwargs['latenttype']
    smeasure=kwargs['similarity']
    plt.figure()
    im = plt.imshow(A)
    plt.colorbar(im)
    plt.title(f'metric={smeasure},cutoff={cutoff}')
    plt.savefig(f'networkdeg_latent{latenttype}_{smeasure}_cutoff{cutoff}.png')
    plt.close()

def plotDegreeHist(A,allDF,nNeighbors=8, **kwargs):
    cutoff = kwargs['cutoff']
    latenttype=kwargs['latenttype']
    smeasure=kwargs['similarity']
    #undirected graph
    degree = np.sum(A, 1)*0.5
    #print names of small degree nodes
    small = np.where(degree<=nNeighbors)[0]
    if not small is None:
        for item in small:
            print (allDF.loc[allDF.index[item], 'gauge_str'])
    plt.figure()
    plt.hist(degree,bins=10)
    plt.savefig(f'networkhist_latent{latenttype}_{smeasure}_cutoff{cutoff}.png')
    plt.close()

def findBadNeighbors(allDF, A, **kwargs):
    """find bad neighborhoods
    Parameters
    ----------
    allDF: df containing static attributes 
    A: adj matrix
    """
    baseline=False
    if baseline:
        #load baseline
        run_res = '../klstm/runs/run_2802_2139_seed789105/lstm_seed789105.p'
        res = pkl.load(open(run_res, 'rb'))

        nselstm=np.zeros(len(res.keys()))
        for ix, item in enumerate(res.keys()): 
            dflstm = res[item]
            nselstm[ix] = Hydrostats.nse(dflstm['qobs'], dflstm['qsim']) 
    else:
        nselstm = pkl.load(open('data/gwnet_nse_statics_nldas_seq30_euclidean_cutoff95_latenttypefull_hiddensize32_seed2251819_log.pkl', 'rb'))        
    G, pos = genGraph(A, allDF)

    nodelist = range(len(nselstm))

    for inode in nodelist:
        nn_nse  = []
        for n in G.neighbors(inode):
            #print (allDF.loc[allDF.index[n], ['gauge_id', 'nse']])
            nn_nse.append(nselstm[n])
        if nselstm[inode]<0.0 and np.mean(nn_nse)<0.0:
            print (f"bad node {inode}, {allDF.loc[allDF.index[inode], ['gauge_str']]}, node nse {nselstm[inode]:.2f}, mean nse {np.mean(nn_nse):.2f}")

def getEnsembleStats():
    """Generate ensemble stat from 10 models
    """
    import glob
    def getStats(runs, modelname='klstm'):
        meanNSE=[]
        medianNSE=[]
        minNSE=[]
        maxNSE=[]
        if modelname=='klstm':
            for runitem in runs:
                run_res = '/'.join([rootdir, runitem, 'lstm_seed{0}.p'.format(runitem[18:])])
                res = pkl.load(open(run_res, 'rb'))

                nselstm=np.zeros(len(res.keys()))
                for ix, item in enumerate(res.keys()): 
                    dflstm = res[item]
                    nselstm[ix] = Hydrostats.nse(dflstm['qsim'],dflstm['qobs']) 
                #get stat
                meanNSE.append(np.mean(nselstm))
                medianNSE.append(np.median(nselstm))
                minNSE.append(np.min(nselstm))
                maxNSE.append(np.max(nselstm))
                print (runitem)
                print (f"""mean nse {np.mean(nselstm):.3f}, median {np.median(nselstm):.3f},
                    minNSE {np.min(nselstm):.3f}, maxNSE {np.max(nselstm):.3f}"""
                )
        elif modelname=='gwt':
            for seed in runs:
                nselstm=pkl.load(open('data/gwnet_nse_statics_nldas_seq30_euclidean_cutoff95_latenttypefull_hiddensize32_seed{0}_log.pkl'.format(seed),'rb'))
                #get stat
                meanNSE.append(np.mean(nselstm))
                medianNSE.append(np.median(nselstm))
                minNSE.append(np.min(nselstm))
                maxNSE.append(np.max(nselstm))    
        elif modelname=='impnet':
            for seed in runs:
                nselstm=pkl.load(open('data/impwnetbestmodel_nldas_seq30_full_latent27_euclidean_cut95.0_seed{0}_nu400_statics_log.pth'.format(seed),'rb'))
                #get stat
                meanNSE.append(np.mean(nselstm))
                medianNSE.append(np.median(nselstm))
                minNSE.append(np.min(nselstm))
                maxNSE.append(np.max(nselstm))    

        else:
            raise NotImplementedError('not found')
        print (meanNSE)
        print (medianNSE)
        print (f'mean NSE {np.mean(meanNSE)}')
        print (f'median NSE {np.mean(medianNSE)}')
        print (f'mean minNSE {np.mean(minNSE)}')
        print (f'mean maxNSE {np.mean(maxNSE)}')  

    rootdir='../klstm/runs'
    #runfolders = glob.glob('/'.join([rootdir,'run_*']))
    #runs = [item[item.rfind('/')+1:] for item in runfolders]
    runs = [
        'run_0103_0015_seed268224',  
        'run_0103_0015_seed442298',
        'run_2802_2013_seed308539',       
        'run_2802_2013_seed500858',
        'run_2802_2014_seed422055',
        'run_2802_2039_seed175943',        
        'run_2802_2133_seed673465',
        'run_2802_2138_seed837109',
        'run_2802_2139_seed789105',
        'run_2802_2157_seed289935',
    ]
    getStats(runs,modelname='klstm')

    gwtseeds=[
    '2251819',
    '2251817',
    '2251815',
    '2251813',
    '2251802',
    '2250611',
    '2210901',
    '2210711',
    '2211057',
    ]
    #getStats(gwtseeds,modelname='gwt')

    impnetseeds=range(614595,614604)
    #getStats(impnetseeds, modelname='impnet')
def main():
    #v1 use latent vector, v2 use full vector

    version=2 #1 or 2
    #similarity measure
    #smeasure = 'mahalanobis'
    #smeasure = "correlation"
    smeasure = "euclidean"

    pnt_cutoff = 99
    allDF = getStaticAttr()

    if version==1:
        latent_dim = 8
        df = getSubSet(allDF)
        print ('use latent vector for adjacency matrix')
        latentMat = train(df, reTrain=False, latent_dim=latent_dim)
    else:
        df = getSubSet(allDF)
        latent_dim = df.shape[1]
        print ('use full attribute vector for adjacency matrix')
        print (df.columns)
        latentMat = df.to_numpy()

    #save the latent mat
    np.save(f'data/latentMat_dim{latent_dim}_v{version}',latentMat)
    A,D = genAdjacency(latentMat,perc=pnt_cutoff, smeasure=smeasure)
    G,pos = genGraph(A, allDF)
    drawGraph(G,pos, nodelist=[288, 303, 342, 385, 410])

    #save the adjacency matrix
    filename = f'data/camels_adaj_{smeasure}_latent{latent_dim}_cutoff{pnt_cutoff}.npz'
    filename1 = f'data/camels_edge_{smeasure}_latent{latent_dim}_cutoff{pnt_cutoff}.npz'
    scipy.sparse.save_npz(filename, netx.to_scipy_sparse_matrix(G))
    scipy.sparse.save_npz(filename1, scipy.sparse.coo_matrix(D))
    sys.exit()
    kwargs={
        'cutoff':pnt_cutoff,
        'latenttype':latent_dim,
        'similarity':smeasure,
    }
    #plot histogram 
    #plotDegreeHist(A,allDF,8, **kwargs)
    #find bad neighbood using baseline
    #findBadNeighbors(allDF, A)
    getEnsembleStats()

    #plotDegreeMap(A,**kwargs)
    doCluster=False
    #clustering on encoded features
    if doCluster:
        n_clusters=10
        kmeans = KMeans(n_clusters=n_clusters)
        clustered_training_set = kmeans.fit_predict(latentMat)
        plotStations(allDF,clustered_training_set,n_clusters)

if __name__ == '__main__':
    main()
