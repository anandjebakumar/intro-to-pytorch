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
from torch.nn import CrossEntropyLoss
from torch.nn import Conv2d
from torch.nn import MaxPool2d

from torch.optim import SGD
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from matplotlib import pyplot as plt
from math import floor
from numpy import vstack
from numpy import argmax
import time

manual_seed(0) # to get reproducible results

class CNN(Module):
    def __init__(self,in_size,num_in_ch,arch_parameters,out_size):
        super().__init__()
        # hard coded for 2 layers each with a conv, relu and maxpool
        # and a fully connected layer 3
        # no padding for any of the layers
        layer1, layer2, layer3 = arch_parameters
        param_conv1, param_pool1 = layer1
        param_conv2, param_pool2 = layer2
        out_ch1, ker_size1c, stride1c = param_conv1
        ker_size1p, stride1p          = param_pool1
        out_ch2, ker_size2c, stride2c = param_conv2
        ker_size2p, stride2p          = param_pool2
        in_ch1 = num_in_ch
        in_ch2 = out_ch1
        out_size1c = self.getOutSize(in_size,ker_size1c,stride1c)
        out_size1p = self.getOutSize(out_size1c,ker_size1p,stride1p)
        out_size2c = self.getOutSize(out_size1p,ker_size2c,stride2c)
        out_size2p = self.getOutSize(out_size2c,ker_size2p,stride2p)
        self.flatten_size = out_ch2*out_size2p**2
        num_hidden_units = layer3
        # layer 1
        self.conv1 = Conv2d(in_ch1,out_ch1,ker_size1c,stride1c)
        self.act1  = ReLU()
        self.pool1 = MaxPool2d(ker_size1p,stride1p)
        # layer 2
        self.conv2 = Conv2d(in_ch2,out_ch2,ker_size2c,stride2c)
        self.act2  = ReLU()
        self.pool2 = MaxPool2d(ker_size2p,stride2p)
        # layer 3
        self.hidden3 = Linear(self.flatten_size,num_hidden_units)
        self.act3    = ReLU()
        # layer 4
        self.hidden4 = Linear(num_hidden_units,out_size)
        self.act4    = Softmax(dim=1)

    def getOutSize(self,in_size,ker_size,stride,pad=0):
        return floor((in_size-ker_size+2*pad)/stride)+1
    
    def forward(self,X):
        # layer 1
        X = self.conv1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # layer 2
        X = self.conv2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # layer 3 - flatten first
        X = X.view(-1,self.flatten_size)
        X = self.hidden3(X)
        X = self.act3(X)
        # layer 4
        X = self.hidden4(X)
        X = self.act4(X)
        return X
        
    def train_model(self, training_set):
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(),lr=0.01,momentum=0.9)
        for epoch in range(1):
            for i, (inputs, targets) in enumerate(training_set):
                optimizer.zero_grad()
                yhat = self.forward(inputs)
                loss = criterion(yhat,targets)
                loss.backward()
                optimizer.step()
            print('finished epoch %d' %epoch)
            
    def evaluate_model(self,dev_set):
        predictions = []
        actuals = []
        for i, (inputs,targets) in enumerate(dev_set):
            yhat = self.forward(inputs)
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            yhat = argmax(yhat,axis=1)       
            actual = actual.reshape((len(actual),1)) # shape initially is (len(actual),) - funny numpy thing
            yhat = yhat.reshape((len(yhat),1))
            predictions.append(yhat)
            actuals.append(actual)
        predictions = vstack(predictions) # to stack everything in a column
        actuals = vstack(actuals)
        acc = accuracy_score(actuals,predictions)
        return acc, predictions
            
def prepare_data(path):
    trans = ToTensor()
    train = MNIST(path,train=True,download=True,transform=trans)
    dev = MNIST(path,train=False,download=True,transform=trans)
    training_set = DataLoader(train,batch_size=32,shuffle=True)
    dev_set = DataLoader(dev,batch_size=32,shuffle=True)
    return training_set, dev_set
         
     
in_size = 28
num_in_ch = 1
param_conv1 = (32,3,1)
param_pool1 =    (2,2)
param_conv2 = (32,3,1)
param_pool2 =    (2,2)
num_hidden_units = 100
layer1 = (param_conv1, param_pool1)
layer2 = (param_conv2, param_pool2)
layer3 = num_hidden_units
arch_parameters = (layer1,layer2,layer3)
out_size = 10
training_set,dev_set = prepare_data('./data/')
print('size of training set %d' %(len(training_set.dataset)))
print('size of dev      set %d' %(len(dev_set.dataset)))
net = CNN(in_size,num_in_ch,arch_parameters,out_size)
tic = time.time()
net.train_model(training_set)
toc = time.time()
print('training time = %3.2f s' %(toc-tic))
tic = time.time()
acc, predictions = net.evaluate_model(dev_set)
toc = time.time()
print('testing time = %3.2f s' %(toc-tic))
print('Accuracy = %3.2f%%' %(100*acc))
