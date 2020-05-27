# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:44:16 2020

@author: Anand Jebakumar
"""

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import manual_seed
manual_seed(0) # to get reproducible results
np.random.seed(0)

class customDataset(Dataset):
    def __init__(self,size):
        self.array = np.random.rand(size)

    def __len__(self):
        return(len(self.array))
    
    def __getitem__(self,idx):
        return(self.array[idx])
    
    def getSplits(self,n_dev=0.3):
        dev_size   = round(n_dev*len(self))
        train_size = len(self)-dev_size
        return random_split(self, [train_size, dev_size])
    
testData = customDataset(10)
print(testData.array)

# split randomly into train and dev set
train, dev = testData.getSplits(0.4) # 0.4 denotes the fraction of dev set; default 0.3

# training set
train_dl = DataLoader(train,batch_size=1,shuffle=False)
print('training set')
for i, batch in enumerate(train_dl):
    print(i,batch)

dev_dl = DataLoader(dev,batch_size=1,shuffle=False)
print('dev set')
for i, batch in enumerate(dev_dl):
    print(i,batch)