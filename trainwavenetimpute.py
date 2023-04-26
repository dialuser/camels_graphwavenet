#author: alex sun
#desc: train graphwavenet
#date: 02/22/2021
#adapt for camels
#02/25, modified for missing data imputation
#03/03, fixing dynamic array A
#03/31, revised using GWNet from Wu's original paer
#04/03, fixed bug in cv part (set does not guarantee list order, reverse back to list)
#       did final production run, use batch runimpnet
#       To retrain, use slurm file, batchwaveimpnet
#=============================================================================
from __future__ import division

import torch
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random
import pandas as pd
import argparse
import sys, os,time
import scipy
import pickle as pkl

from readcamels import getStaticAttr
from utils_wnet import load_adj
#this is Wu's original model
from gwnetmodel_impute import GWNet
from util_gtnet import Optim 
from sklearn.model_selection import KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args(args):
    '''Parse training options user can specify in command line.
    Specify hyper parameters here

    Returns
    -------
    argparse.Namespace
        the output parser object
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments used when training GraphWavenetImpute model.",
    )

    # optional input parameters
    parser.add_argument(
        '--n_o',type=int,default=477,
        help='number of observable locations'
    )
    # this is not used?
    parser.add_argument(
        '--n_m',type=int,default=53,
        help='number of mask node during training'
    )

    parser.add_argument(
        '--n_u',type=int,default=53,
        help='target locations, n_u locations will be deleted from the training data'
    )
    #>>>>>GWNet configurations
    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
    parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
    parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
    parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    parser.add_argument('--apt_size', default=10, type=int)
    
    parser.add_argument('--seq_length',type=int,default=30,help='')
    parser.add_argument('--nhid',type=int,default=32,help='')
    parser.add_argument('--in_dim',type=int,default=32,help='inputs dimension')
    parser.add_argument('--num_nodes',type=int,default=530,help='number of nodes')
    parser.add_argument('--batch_size',type=int,default=30,help='batch size')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--print_every',type=int,default=50,help='')
    parser.add_argument("--clipnorm", type=float, default=2.0, help="clip norm")
    #<<<<<<<<

    parser.add_argument('--nepoch',type=int,default=50,help='max training episode')
    parser.add_argument('--learning_rate',type=float,default=0.0001,help='the learning_rate for Adam optimizer')
    parser.add_argument("--L1Loss", action="store_true", default=False, help="true to use L1Loss function")

    #add additional arguments
    parser.add_argument("--retrain",action='store_true', default=False, 
                        help="retrain the model only if true" )
    parser.add_argument("--addstatics",action='store_true', default=False, help="true to include static attributes")
    parser.add_argument("--runtesting",action='store_true', default=False, help="true to run test")
    parser.add_argument("--forcingtype",type=str,default="nldas", help="forcing data to use")
    parser.add_argument("--usefinal",action='store_true', default=False, help="true to use the final saved model; otherwise, use best model")
    parser.add_argument("--latenttype",type=str,default="full", help="size of latent")
    parser.add_argument("--netcutoff",type=float,default=98, help="network cutoff")
    parser.add_argument("--similarity",type=str,default="euclidean", help="similarity measure")
    parser.add_argument("--netlatent",type=int,default=6, help="hidden dim of camel")
    parser.add_argument("--seed", type=int, default=20210221, help="random seed") 
    parser.add_argument("--uselog", action="store_true", default=False, help="true to use log transform")

    parser.add_argument("--kfold", type=int, default=12, help="number of folds") 
    parser.add_argument('--cv', action='store_true', default=False, help='true to do kfold cv')

    return parser.parse_known_args(args)[0]

def getDataLoaders(batchsize,seq,forcingType,latentType,addStatics=False,uselog=False):
    """Configure pytorch dataloaders
    Param
    -----
    uselog: if True, load log-transformed Q data
    
    Returns
    ------
    traing, validation, and testing dataloaders
    nfeature: number of features
    """
    if uselog:
        #from readcamels_log_long import genLSTMDataSets
        from readcamels_log import genLSTMDataSets
    else:
        #from readcamelslong import genLSTMDataSets
        from readcamels import genLSTMDataSets

    trainDataset,valDataset,testDataset,nfeatures = genLSTMDataSets(forcingType=forcingType, 
                    latentType=latentType,
                    seq=seq,
                    addStatics=addStatics)

    trainLoader = DataLoader(trainDataset, batch_size=batchsize, shuffle=True, drop_last=True,num_workers=4)
    valLoader = DataLoader(valDataset, batch_size=batchsize, shuffle=True, drop_last=True,num_workers=4)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, drop_last=True)

    return trainLoader,valLoader,testLoader,nfeatures 

def test(args, model,testLoader,df,A,unobsmask,regen=True):
    """Testing
    Params
    ------
    model: trained model
    testLoader: test data
    df: dataframe from getStatAttributes()
    observable_set: observable node set
    """
    import hydrostats as Hydrostats

    forcingType=args.forcingtype
    seq = args.seq_length
    cutoff = args.netcutoff
    addstatics=args.addstatics
    latenttype=args.latenttype
    smeasure = args.similarity
    uselog = args.uselog
    seed = args.seed
    
    #convert set to list
    nNode = len(unobsmask)    
    print (unobsmask)
    if regen:
        model.eval()
        print ('test data temporal length ', len(testLoader))

        outMat = np.zeros((len(testLoader), nNode))
        trueMat = np.zeros((len(testLoader), nNode))
        if uselog:
            myscaler = pkl.load(open(f'data/camels_forcing_scaler_{forcingType}_log.pkl', 'rb'))  
        else:
            myscaler = pkl.load(open(f'data/camels_forcing_scaler_{forcingType}.pkl', 'rb'))  

        idex=0
        #0403, note must use full adj, don't try to do partial A
        #normalize the full adj matrix
        adj_mx = load_adj(A, args.adjtype)

        if args.aptonly:
            supports = None 
        else:
            supports = [torch.tensor(i).to(device) for i in adj_mx]

        model.supports = supports

        for x,y in testLoader:
            x = x.to(device)
            with torch.no_grad():
                out = model(x).squeeze()            
            out = out.data.cpu().numpy()

            y = y.data.cpu().numpy()
            if uselog:
                y = np.exp(y*myscaler['output_std']+myscaler['output_mean'])
                out = np.exp(out*myscaler['output_std']+myscaler['output_mean'])
            else:
                y = y*myscaler['output_std']+myscaler['output_mean']
                out = out*myscaler['output_std']+myscaler['output_mean']

            out[out<0]=0.0
            y[y<0.0] = 0.0
            #one step at a time
            #extract results at unobs locations
            outMat[idex,:]  = out.flatten()[unobsmask]
            trueMat[idex,:] = y.flatten()[unobsmask]
            idex+=1
        #calculate nse on missing nodes
        nse=np.zeros((nNode))
        for i in range(nNode):
            df0 = pd.DataFrame(np.c_[outMat[:,i],trueMat[:,i]],columns=('qsim','qobs'))
            df0 = df0.dropna()
            nse[i]= Hydrostats.nse(df0['qsim'], df0['qobs']) 
    print (f'median nse {np.median(nse):.3f}, mean nse {np.mean(nse):.3f}, max nse, {np.max(nse):.3f}, min nse, {np.min(nse):.3f}')

    return trueMat,outMat

def getNodeSets(A, n_u,unknown_set=None):
    """configure node sets for training
    Params
    ------
    A: full adjacency matrix
    n_u: number of unobserved nodes
    Returns
    ------
    known_set: observable nodes [note this should be greater than n_o for random sampling]
    unknown_set: unmeasured nodes
    full_set: all node set
    """
    num_nodes = A.shape[0]
    if unknown_set is None:
        unknown_set = set(np.random.choice(list(range(0,num_nodes)),n_u,replace=False))
    
    full_set = set(range(0,num_nodes))        
    known_set = full_set - unknown_set
    return known_set,unknown_set,full_set

def trainEpoch(model,optimizer,loader,criterion,observable_set):
    """Train a single epoch
    Params
    ------
    model: model to be trained
    A: full adj matrix
    optimizer: optimizer
    loader: training dataloader
    criterion: loss function
    observable_set: list of observable nodes to be sampled 
    n_o: number of nodes to be sampled (<len(observable_set))
    """
    model.train()
    clip_norm = True
    n=0
    l_sum=0.0

    mask = observable_set 

    for x,y in loader:         
        x=x[:,:,mask,:] #extract the training nodes only
        y=y[:,mask]

        x=x.to(device) #(batch_size, input_seq_length, num_nodes, input_feature_dim)
        y=y.to(device) #(batch, nnode)    
        model.zero_grad()  # Clear gradients.
        out = model(x,mask).squeeze() #(batch, nnode)
        
        loss = criterion(out, y.detach())  # Compute the loss solely based on the training nodes.
        l_sum +=loss.item()
        n+=out.shape[0]

        loss.backward()  # Derive gradients.
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipnorm)
        optimizer.step()  # Update parameters based on gradients.

    return l_sum/n,mask

def evalEpoch(model,loader,criterion,mask):
    """get validation stat on a single epoch
    Params
    -----
    mask: the randomly generated observable mask passed from trainEpoch()
    A_q: the dynamic adj matrix passed from trainEpoch()
    """
    model.eval()
    n=0
    l_sum=0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        #extract training locations
        x=x[:,:,mask,:]
        y=y[:,mask]
        with torch.no_grad():
            out = model(x,mask).squeeze()  # Perform a single forward pass.

        #out=out[:,mask]
        #y = y[:,mask]
        loss = criterion(out, y)  # Compute the loss solely based on the training nodes.
        l_sum +=loss.item()
        n+=out.shape[0]
    return l_sum/n    

def train(args, trainLoader, valLoader, A, in_dim, observable_set,
        save_path, out_dim = 1, reTrain=False,kfold=0):
    """Train the model
    Parameters
    ----------
    trainLoader: training dataset
    valLoader: validation dataset
    A: full adj matrix
    in_dim: feature dimension
    observable_set, set of all observable nodes
    out_dim, temporal length of model prediction
    save_path: path to output model file
    out_dim: number of future forecast steps
    """
    addaptadj = args.addaptadj

    print (f"in feature dim {in_dim}, out feature dim {out_dim}")
    num_nodes=A.shape[0] 

    n_o = args.n_o
    nepoch = args.nepoch
    learning_rate = args.learning_rate
    addStatics = args.addstatics
    forcingType = args.forcingtype
    seq = args.seq_length
    latentsize = args.netlatent
    similarity = args.similarity
    netcutoff = int(args.netcutoff)
    latentType =args.latenttype
    usefinal = args.usefinal
    uselog = args.uselog
    seed = args.seed

    observable_set = list(observable_set)
    basestr = f'{forcingType}_seq{seq}_{latentType}_latent{latentsize}_{similarity}_cut{netcutoff}_seed{seed}_{kfold}_nu{n_o}' 
    if addStatics:
        if uselog:
            model_path='/'.join([save_path,f'impwnetbestmodel_{basestr}_statics_log.pth'])
            model_path_finale='/'.join([save_path,f'impwnetfinalmodel_{basestr}_statics_log.pth'])
        else:
            model_path='/'.join([save_path,f'impwnetbestmodel_{basestr}_statics.pth'])
            model_path_finale='/'.join([save_path,f'impwnetfinalmodel_{basestr}_statics.pth'])
    else:
        if uselog:
            model_path='/'.join([save_path,f'impwnetbestmodel_{basestr}_log.pth'])
            model_path_finale='/'.join([save_path,f'impwnetfinalmodel_{basestr}_log.pth'])
        else:
            model_path='/'.join([save_path,f'impwnetbestmodel_{basestr}.pth'])
            model_path_finale='/'.join([save_path,f'impwnetfinalmodel_{basestr}.pth'])

    A = A[observable_set,:][:,observable_set]
    adj_mx = load_adj(A, args.adjtype)
    #eqn 7 in the paper
    if args.aptonly:
        supports = None 
    else:
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        print ('support len', len(supports))

    # Define model
    model = GWNet(device, 
        num_nodes=num_nodes, 
        dropout=args.dropout,
        supports=supports, 
        gcn_bool=args.gcn_bool, 
        addaptadj=addaptadj, 
        aptinit=None, 
        in_dim=in_dim, 
        out_dim=out_dim, 
        residual_channels=args.nhid, 
        dilation_channels=args.nhid, 
        skip_channels=args.nhid * 4, #original *8 
        end_channels=args.nhid * 8, #original *16
        apt_size = args.apt_size,
        kernel_size=4) #must change kernel_size to 4 to make this work

    model = model.to(device)

    if reTrain:
        if args.L1Loss:
            criterion = torch.nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        optimizer = Optim(model.parameters(), 'adam', args.learning_rate, 1.0, 
                lr_decay=args.weight_decay, start_decay_at=20)

        min_val_loss = np.Inf
        for epoch in range(nepoch):
            epochTrainLoss,mask = trainEpoch(model,optimizer,trainLoader,criterion,observable_set)
            epochValLoss = evalEpoch(model,valLoader,criterion,mask)
            print("epoch", epoch, ", train loss:", epochTrainLoss, ", val loss:",epochValLoss)
            if epochValLoss < min_val_loss:
                min_val_loss = epochValLoss
                if epoch>10:
                    torch.save(model.state_dict(), model_path)

        #save the final model
        torch.save(model.state_dict(), model_path_finale)
    
    if not usefinal:
        print ('use saved best model ', model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print ('use saved final model ', model_path_finale)
        model.load_state_dict(torch.load(model_path_finale))
    return model


def main(args):
    """
    Model training
    """
    #original parameters
    n_m = args.n_m #number of missing data
    n_u = args.n_u #number of unknown targets
    batch_size = args.batch_size

    #fix random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    save_path = 'models'
    netcutoff=int(args.netcutoff) #network cutoff in terms of percent
    addstatics=args.addstatics
    forcingType=args.forcingtype
    latentType =args.latenttype
    smeasure = args.similarity
    seq = args.seq_length
    latentsize=args.netlatent #latent dim used for camels, this is legacy parameter

    #load pre-defined adj matrix
    adjfile = f'data/camels_adaj_{smeasure}_latent{latentsize}_cutoff{netcutoff}.npz'
    A = scipy.sparse.load_npz(adjfile)
    
    #load dataset
    trainLoader,valLoader,testLoader,nfeatures = getDataLoaders(
        batchsize=batch_size, 
        seq=seq, 
        addStatics=addstatics,
        forcingType=forcingType,
        latentType=latentType,
        uselog=args.uselog,    
    )
    df = getStaticAttr()

    # node configurations
    if args.cv and args.retrain:
        #do k-fold split by fixing the seed
        kf = KFold(n_splits=args.kfold, random_state=args.seed, shuffle=True)
        counter=0
        for train_index, test_index in kf.split(np.arange(A.shape[0])):
            unknown_set = set(test_index)
            print ('unknown_set', unknown_set)
            n_u = len(unknown_set)
            observable_set,unknown_set,full_set = getNodeSets(A,n_u,unknown_set)
            model = train(args, trainLoader,valLoader,A, nfeatures,observable_set, 
                save_path=save_path, reTrain=args.retrain,kfold=counter )
            #print per fold NSE for testin purposes, the final result is saved in the block below                
            obscv, simcv =test(args, model, testLoader, df=df, A=A, 
                unobsmask=test_index)
            counter+=1

    if args.runtesting:
        if args.cv:
            kf = KFold(n_splits=args.kfold, random_state=args.seed, shuffle=True)
            counter=0
            simMat = np.zeros((len(testLoader), df.shape[0]))
            obsMat = np.zeros((len(testLoader), df.shape[0]))
            nnodes=0
            for train_index, test_index in kf.split(np.arange(A.shape[0])):
                unknown_set = set(test_index)
                observable_set,unknown_set,full_set = getNodeSets(A,n_u,unknown_set)
                
                #reload the model for different fold!!!!
                model = train(args, trainLoader,valLoader,A, nfeatures,observable_set, 
                      save_path=save_path, reTrain=False,kfold=counter )

                obscv, simcv = test(args, model, testLoader, df=df, A=A, 
                            unobsmask=test_index)
                simMat[:,test_index] = simcv
                obsMat[:,test_index] = obscv
                #save the result
                print (f'finished fold{counter}')
                counter+=1
                nnodes +=len(test_index)
            print ('total nodes processd ', nnodes)
            pkl.dump([obsMat,simMat], open(f"cvwaveres/impexp_{args.seed}.pkl", 'wb'))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print (args)
    main(args)