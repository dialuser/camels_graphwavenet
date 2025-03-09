#https://gist.githubusercontent.com/branislav1991/4c143394bdad612883d148e0617bdccd/raw/aa0d5a46cdfd418fa9e6ccf12e15de729deb408b/hdf5_dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

def toTensor(nparr, dtype):
    """utility function for tensor conversion
    """
    return torch.tensor(nparr, dtype=dtype)

class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Params:
    ------
    h5file_name, name of the hdf5 file
    seq, sequence length
    statics_file, file holding static feature
    """
    def __init__(self, h5file_name, seq, statics_file=None,lazy_load=True):
        super().__init__()
        
        #lazy loading
        hf = h5py.File(h5file_name, 'r')
        if lazy_load:
            self.XDS = hf['X']
            self.YDS = hf['Y']
        else:
            self.XDS = hf['X'][:]
            self.YDS = hf['Y'][:]

        self.datalen = self.XDS.shape[0]
        if not statics_file is None:
            self.featuremat = self.__getStatics(seq,statics_file)            
            self.addStatics = True
        else:
            self.addStatics = False

    def __getitem__(self, index):
        # get data        
        x = self.XDS[index]
        if self.addStatics:
            x = np.concatenate((x, self.featuremat), axis=-1)
        x = toTensor(x, dtype=torch.float32)
        # get label
        y = self.YDS[index]
        y = toTensor(y, dtype=torch.float32)
        
        return (x, y)

    def __len__(self):
        #the len() of a dataset is the length of the first axis
        return  self.datalen
    
    def __getStatics(self,seq,statics_file):
        df =  pd.read_pickle(statics_file)
        nBasin = df.shape[0]
        #(seq,nbasin,nfeature)
        featuremat = np.zeros((seq,nBasin,27))
        for ibasin in range(nBasin):
            featuremat[:,ibasin,:]= df.iloc[ibasin,:].to_numpy()[np.newaxis,:].repeat(seq,axis=0)
        #featureTensor = torch.from_numpy(featuremat)            
        return featuremat


def genLSTMDataSets(forcingType, latentType, splitratio=(0.8,0.2), seq=60, addStatics=False,lazy_load=True):
    h5file = f'data/365/camels_lstmlist_{forcingType}_seq{seq}train.h5'    
    statics_file = 'data/365/statics.pkl'
    
    trainDataset = HDF5Dataset(h5file,seq=seq,statics_file=statics_file,lazy_load=lazy_load)

    h5file = f'data/365/camels_lstmlist_{forcingType}_seq{seq}val.h5'    
    valDataset = HDF5Dataset(h5file,seq=seq,statics_file=statics_file,lazy_load=lazy_load)

    h5file = f'data/365/camels_lstmlist_{forcingType}_seq{seq}test.h5'    
    testDataset = HDF5Dataset(h5file,seq=seq,statics_file=statics_file,lazy_load=lazy_load)
    
    nfeatures = 32
    return trainDataset,valDataset,testDataset,nfeatures

def main():
    h5file = 'data/365/camels_lstmlist_nldas_seq120val.h5'
    statics_file = 'data/365/statics.pkl'
    startTime = time.time()
    dataset = HDF5Dataset(h5file,seq=120,statics_file=statics_file,lazy_load=False)
    print ('loading time', time.time()-startTime)
    dataloader = DataLoader(dataset, batch_size=100, 
                        shuffle=True,drop_last=False,num_workers=4,
                        pin_memory=True)    
    startTime = time.time()
    for idx, (x,y) in enumerate(dataloader):
        print (x.shape)
    print ('loop time', time.time()-startTime)
if __name__ == '__main__':
    main()