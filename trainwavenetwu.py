#author: alex sun
#date: 02152021
#date: 03012021, finalize for production runs
#date: 03032021, replace the optimizer
#date: 03292021, use the original graphwavenet by wu
#date: 0403: used for final production run
#use batch runwave2 to run all cases
#=============================================================================
import random 
import torch
from torch.utils.data import DataLoader

import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from torch.autograd import Variable
import scipy.sparse as sp
import argparse
import sys
from gwnetmodel import GWNet
from utils_wnet import load_adj

from readcamels import getStaticAttr
from util_gtnet import Optim 
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_shared_arg_parser():
    """set the default parameters
    """
    parser = argparse.ArgumentParser()
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

    return parser

def getDataLoaders(batchsize,seq,forcingType,latentType,addStatics=False,uselog=False):
    if uselog:
        from readcamels_log import genLSTMDataSets
    else:
        from readcamels import genLSTMDataSets

    trainDataset,valDataset,testDataset,nfeatures = genLSTMDataSets(forcingType=forcingType, 
                    latentType=latentType,
                    seq=seq,
                    addStatics=addStatics)

    trainLoader = DataLoader(trainDataset, batch_size=batchsize, shuffle=True, drop_last=True,num_workers=4)
    valLoader = DataLoader(valDataset, batch_size=batchsize, shuffle=True, drop_last=True,num_workers=4)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, drop_last=True)

    return trainLoader,valLoader,testLoader,nfeatures 

def trainEpoch(model,optimizer,loader,criterion,args):
    model.train()
    clip_norm = True
    n=0
    l_sum=0.0
    starttime = time.time()
    for x,y in loader: 
        x=x.to(device)     
        y=y.to(device)     #(batch, nnode)    
        model.zero_grad()  # Clear gradients.
        out = model(x).squeeze() #(batch, nnode)

        loss = criterion(out, y.detach())  # Compute the loss solely based on the training nodes.

        l_sum +=loss.item()
        n+=out.shape[0]
        loss.backward()  # Derive gradients.
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipnorm)
        optimizer.step()  # Update parameters based on gradients.
    print ('time elapsed ', time.time()-starttime)
    return l_sum/n    

def evalEpoch(model,loader,criterion):
    model.eval()
    n=0
    l_sum=0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x).squeeze()  # Perform a single forward pass.

        loss = criterion(out, y)  # Compute the loss solely based on the training nodes.
        l_sum +=loss.item()
        n+=out.shape[0]
    return l_sum/n    

def test(args,model,testLoader,df,regen):
    import hydrostats as Hydrostats

    forcingType=args.forcingtype
    seq = args.seq_length
    cutoff = int(args.netcutoff)
    addstatics=args.addstatics
    latenttype=args.latenttype
    hiddensize=args.hiddensize
    smeasure = args.similarity
    uselog = args.uselog
    seed = args.seed

    nNode = df.shape[0]

    if regen:
        model.eval()
        print ('number of test data', len(testLoader))

        outMat = np.zeros((len(testLoader), nNode))
        trueMat = np.zeros((len(testLoader), nNode))
        if uselog:
            myscaler = pkl.load(open(f'data/camels_forcing_scaler_{forcingType}_log.pkl', 'rb'))  
        else:
            myscaler = pkl.load(open(f'data/camels_forcing_scaler_{forcingType}.pkl', 'rb'))  

        nse=np.zeros((nNode,))
        idex=0
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
            
            outMat[idex,:]=out
            trueMat[idex,:] = y
            idex+=1

        for i in range(nNode):
            df =pd.DataFrame(np.c_[outMat[:,i],trueMat[:,i]], columns=('qsim','qobs'))
            df = df.dropna()            
            nse[i]= Hydrostats.nse(df['qsim'],df['qobs'])
            
    print (f'median nse {np.median(nse):.3f}, mean nse {np.mean(nse):.3f}, max nse {np.max(nse):.3f}, min nse {np.min(nse):.3f}')

    return trueMat,outMat

def train(args, trainLoader, valLoader, A, in_dim, save_path, 
        out_dim=1, reTrain=False):
    """
    Parameters:
    ---------
    num_nodes: number of graph nodes
    in_dim: feature dim of input
    out_dim: number of prediction steps  (prediction length, t+1, t+2,...)
    addaptadj: whether to add apt adj matrix (eq 6) to graph conv layer
    apt_size: size of latent dim for randomly initializing node embedding

    """    

    num_nodes = A.shape[0]
    
    nEpoch = args.nepoch
    lr = args.learnrate
    dropout = args.dropout
    addStatics = args.addstatics
    forcingType= args.forcingtype
    seq = args.seq_length
    latentsize = args.netlatent
    similarity = args.similarity
    netcutoff = args.netcutoff
    latentType =args.latenttype
    usefinal = args.usefinal
    uselog = args.uselog
    addaptadj = args.addaptadj

    print (in_dim, out_dim)
    adj_mx = load_adj(A, args.adjtype)

    if args.aptonly:
        supports = None 
    else:
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        print ('support len', len(supports))

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

    seed = args.seed
    if args.L1Loss:
        lossfunstr = 'L1'
    else:
        lossfunstr = "L2"
    basestr = f'{forcingType}_seq{seq}_{latentType}_latent{latentsize}_{similarity}_cut{int(netcutoff)}_{lossfunstr}_seed{seed}'
    if addStatics:
        if uselog:
            model_path='/'.join([save_path, f'gwnet2bestmodel_{basestr}_statics_log.pth'])
            model_path_finale='/'.join([save_path,f'gwnet2finalmodel_{basestr}_statics_log.pth'])
        else:
            model_path='/'.join([save_path,f'gwnet2bestmodel_{basestr}_statics.pth'])
            model_path_finale='/'.join([save_path,f'gwnet2finalmodel_{basestr}_statics.pth'])
    else:
        if uselog:
            model_path='/'.join([save_path,f'gwnet2bestmodel_{basestr}_log.pth'])
            model_path_finale='/'.join([save_path,f'gwnet2finalmodel_{basestr}_log.pth'])
        else:
            model_path='/'.join([save_path,f'gwnet2bestmodel_{basestr}.pth'])
            model_path_finale='/'.join([save_path,f'gwnet2finalmodel_{basestr}.pth'])

    model.to(device)    
    if reTrain:
        model.train()
        optimizer = Optim(
                model.parameters(), 'adam', lr, 1.0, 
                lr_decay=1e-5, start_decay_at=20)

        min_val_loss = np.inf
        if args.L1Loss:
            lossfun = torch.nn.L1Loss()
        else:
            lossfun = torch.nn.MSELoss()

        for epoch in range(nEpoch):
            epochTrainLoss = trainEpoch(model,optimizer,trainLoader,lossfun,args)
            epochValLoss = evalEpoch(model,valLoader,lossfun)
            print("epoch", epoch, ", train loss:", epochTrainLoss, ", val loss:",epochValLoss)
            if epochValLoss < min_val_loss:
                min_val_loss = epochValLoss
                if epoch>10:
                    torch.save(model.state_dict(), model_path)

        torch.save(model.state_dict(), model_path_finale)            
    if not usefinal:
        print ('use saved best model ', model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print ('use saved final model ', model_path_finale)
        model.load_state_dict(torch.load(model_path_finale))

    return model
        

def main():
    parser = get_shared_arg_parser() 
    #add additional arguments
    parser.add_argument("--retrain",action='store_true', default=False, 
                        help="retrain the model only if true" )
    parser.add_argument("--nepoch", type=int, default=60, help="set the number of epochs")
    parser.add_argument("--learnrate",type=float,default=0.01, help="learning rate")
    parser.add_argument("--addstatics",action='store_true', default=False, help="true to include static attributes")
    parser.add_argument("--runtesting",action='store_true', default=False, help="true to run test")
    parser.add_argument("--forcingtype",type=str,default="nldas", help="forcing data to use")
    parser.add_argument("--usefinal",action='store_true', default=False, help="true to use the final saved model; otherwise, use best model")
    parser.add_argument("--latenttype",type=str,default="full", help="size of latent")
    parser.add_argument("--netcutoff",type=float,default=95, help="network cutoff")
    parser.add_argument("--similarity",type=str,default="euclidean", help="similarity measure")
    parser.add_argument("--netlatent",type=int,default=6, help="hidden dim of camel")
    parser.add_argument('--hiddensize', type=int, default=64, help='hidden size')
    parser.add_argument("--seed", type=int, default=20210221, help="random seed") 
    parser.add_argument("--uselog", action="store_true", default=False, help="true to use log transform")
    parser.add_argument("--L1Loss", action="store_true", default=False, help="true to use L1Loss function")

    opt = parser.parse_args()
    print ('options ', opt)
    #
    #set random seed    
    #
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    save_path = 'models'
    netcutoff=int(opt.netcutoff) #network cutoff
    addstatics=opt.addstatics
    forcingType=opt.forcingtype
    latentType =opt.latenttype
    smeasure = opt.similarity
    seq = opt.seq_length
    latentsize=opt.netlatent #latent dim used for camels, this is legacy parameter

    trainLoader,valLoader,testLoader,nfeatures = getDataLoaders(
        batchsize=opt.batch_size, 
        seq=seq, 
        addStatics=addstatics,
        forcingType=forcingType,
        latentType=latentType,
        uselog=opt.uselog,    
    )
    adjfile = f'data/camels_adaj_{smeasure}_latent{latentsize}_cutoff{netcutoff}.npz'
    edgefile = f'data/camels_edge_{smeasure}_latent{latentsize}_cutoff{netcutoff}.npz'
    print ('loading adj mat', adjfile)
    print ('loading wt mat', edgefile)
    A = scipy.sparse.load_npz(adjfile)

    model = train(opt, trainLoader,valLoader,A, nfeatures, save_path=save_path, reTrain=opt.retrain)
    if opt.runtesting:
        df = getStaticAttr()
        trueMat,simMat =test(opt, model, testLoader, df=df, regen=True)
        if opt.uselog:
            pkl.dump([trueMat,simMat], open(f"gwnet2res/gwnet2_{opt.seed}_seq{seq}_cutoff{netcutoff}_log.pkl", 'wb'))
        else:
            pkl.dump([trueMat,simMat], open(f"gwnet2res/gwnet2_{opt.seed}_seq{seq}_cutoff{netcutoff}.pkl", 'wb'))


if __name__ == '__main__':
    main()
