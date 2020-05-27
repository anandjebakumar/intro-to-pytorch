# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:44:16 2020

@author: Anand Jebakumar
"""

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

np.random.seed(0)

class customDataset(Dataset):
    def __init__(self,size):
        self.array = np.random.rand(size)

    def __len__(self):
        return(len(self.array))
    
    def __getitem__(self,idx):
        return(self.array[idx])
    
testData = customDataset(10)

# no shuffling
unshuffledLoader = DataLoader(testData,batch_size=2,shuffle=False)
print(testData.array)

for i, batch in enumerate(unshuffledLoader):
    print(i,batch)
    
# random shuffling
shuffledLoader = DataLoader(testData,batch_size=2,shuffle=True)
print(testData.array)

for i, batch in enumerate(shuffledLoader):
    print(i,batch)