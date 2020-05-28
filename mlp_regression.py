# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:48:59 2020

@author: Anand Jebakumar
"""
import torch
import numpy as np
from torch import manual_seed
from torch import tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Softmax
from torch.nn import ReLU
from torch.nn import MSELoss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
manual_seed(0) # to get reproducible results

class CSVDataset(Dataset):
    def __init__(self,fileName):
        df = pd.read_csv(fileName, header=None)
        self.X = tensor(df.values[:,:-1].astype('float32'))
        self.y = tensor(df.values[:,-1].astype('float32'))        
        self.y = self.y.reshape((len(self.y),1))
        self.numInFeatures = len(self.X[0])
        self.numData = len(self.X)
        
    def __len__(self):
        return self.numData
    
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
    def getSplits(self,n_dev=0.3):
        dev_size   = round(n_dev*self.numData)
        train_size = self.numData-dev_size
        train, dev = random_split(self, [train_size, dev_size])
        # not being shuffled since the random split already shuffled
        training_set = DataLoader(train, batch_size = 32, shuffle=False) 
        # dev set is not split into batches
        dev_set = DataLoader(dev, batch_size = dev_size, shuffle=False) 
        return training_set, dev_set
    
class MLP(Module):
    def __init__(self,inputs):
        super().__init__()
        self.hidden1 = Linear(inputs,10) # layer 1 - 10 units
        self.hidden2 = Linear(10,8) # layer 2 - 8 units
        self.output  = Linear(8,1) # layer 3 (output) - 1 unit
        self.relu    = ReLU() # no learnable parameters here
        # He Kaiming initialization
        xavier_uniform_(self.hidden1.weight)
        xavier_uniform_(self.hidden2.weight)
        xavier_uniform_(self.output.weight)
        
    def forward(self, X):
        X = self.hidden1(X) # layer 1 - 10 units 
        X = self.relu(X) # layer 1 - relu activation
        X = self.hidden2(X) # layer 2 - 8 units
        X = self.relu(X) # layer 2 - relu activation
        X = self.output(X) # layer 3 (output) - 1 unit
        return X
    
    def train_model(self, training_set):
        criterion = MSELoss()
        optimizer = SGD(self.parameters(),lr=0.01,momentum=0.9)
        for epoch in range(100): #100 epochs
            for i, (inputs,targets) in enumerate(training_set):
                # clear gradients
                optimizer.zero_grad()
                yhat = self.forward(inputs)
                loss = criterion(yhat,targets)
                loss.backward()
                optimizer.step()
            if(epoch%10 == 0):
                print(epoch,loss)
                
    def evaluate_model(self, dev_set):
        inputs, targets = next(iter(dev_set))
        yhat = self.forward(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy().reshape((len(targets),1))
        acc = mean_squared_error(actual,yhat)
        return acc
        
    def predict(self,row):
        row = tensor([row])
        yhat = self.forward(row)
        yhat = yhat.detach().numpy()
        return yhat
    
dataset = CSVDataset('./data/housing.csv')

net = MLP(dataset.numInFeatures)
out = net(dataset.X)
training_set, dev_set = dataset.getSplits()
net.train_model(training_set)
acc = net.evaluate_model(dev_set)
print(acc)
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = net.predict(row)
print('Predicted: %.3f' % yhat)