# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:48:59 2020

@author: Anand Jebakumar
"""
import torch
from torch import manual_seed
from torch import tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.nn import BCELoss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
manual_seed(0) # to get reproducible results

class CSVDataset(Dataset):
    def __init__(self,fileName):
        df = pd.read_csv(fileName, header=None)
        self.X = tensor(df.values[:,:-1].astype('float32'))
        encoded_output = LabelEncoder().fit_transform(df.values[:,-1])
        self.y = tensor(encoded_output.astype('float32'))
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
        self.sigmoid = Sigmoid() # no learnable parameters here
        # He Kaiming initialization
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        kaiming_uniform_(self.output.weight, nonlinearity='sigmoid')
        
    def forward(self, X):
        X = self.hidden1(X) # layer 1 - 10 units 
        X = self.relu(X) # layer 1 - relu activation
        X = self.hidden2(X) # layer 2 - 8 units
        X = self.relu(X) # layer 2 - relu activation
        X = self.output(X) # layer 3 (output) - 1 unit
        X = self.sigmoid(X) # layer 3 - sigmoid activation
        return X
    
    def train_model(self, training_set):
        criterion = BCELoss()
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
        yhat = yhat.round()
        actual = targets.numpy().reshape((len(targets),1))
        acc = accuracy_score(actual,yhat)
        return acc
        
    def predict(self,row):
        row = tensor([row])
        yhat = self.forward(row)
        yhat = yhat.detach().numpy()
        return yhat
    
dataset = CSVDataset('./data/ionosphere.csv')
net = MLP(dataset.numInFeatures)
out = net(dataset.X)
training_set, dev_set = dataset.getSplits()
net.train_model(training_set)
acc = net.evaluate_model(dev_set)
print(acc)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = net.predict(row)
print(yhat)
