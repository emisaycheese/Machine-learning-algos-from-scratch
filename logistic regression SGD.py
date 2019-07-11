#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:39:54 2019

@author: ruiqianyang
"""

import numpy as np

batch_size=100
steps=1000
alpha=0.5

def sigmoid(z):
    return 1/(1+np.e**(-z))
    
#def logistic_loss(y,y_hat):
#    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

#get prediction
def pred(X,W):
    z=np.dot(X,W)
    return sigmoid(z)[:,0]

def gradient(X,y,y_hat):
    m=X.shape[0]
    return np.dot(X.T,(y_hat-y))/m

start=0
end=batch_size
w=np.random.random([n,1])   #n features
# dataset is:df
# if sgd, we can make steps=100000, batch_size=1


for i in range(steps):
    x_batch,y_batch=df.iloc[start:end,:][features],df.iloc[start:end,:]['y']
    y_hat=pred(x_batch,w)
    dw=gradient(x_batch,y_batch,y_hat)
    w-=alpha*dw
    
    
    start+=batch_size
    end+=batch_size
    alpha*=0.99
#%%
batch_size=1
steps=len(data)
learning_rate=0.5

def sigmoid(z):  
    return 1/(1+np.e**(-z))

def forward(x,w):
    z=np.dot(x,w)
    return sigmoid(z)[:,0]

def gradient(x,y,y_hat):
    m=x.shape[0]
    return (1/m)*np.dot(x.T,y_hat-y)
    
start=0
end=batch_size


for i in range(steps):
    x_batch,y_batch=df.iloc[start:end,:][features_columns],
    y_hat=forward(x_batch,w)
    dw=gradient(x_batch,y_batch,y_hat)  
    w-=learning_rate*dw
    
    start+=batch_size
    end+=batch_size
    learning_rate*=0.99


    
    
#%%
batch_size=100
steps=1000
alpha=0.5


def sigmoid(z):
    return 1/(1+np.e**(-z))

def forward(x,w):
    z=np.dot(x,w)
    return sigmoid(z)[:,0]
    
def gradient(x,y,y_hat):
    m=x.shape()[0]
    return (1/m)*np.dot(x.T,(y_hat-y))    
    
start=0
end=batch_size
    
w=np.random.random([n_freatures,1])
for i in range(steps):
    x,y=df.iloc[start:end,features_col],df.iloc[start:end,'y']
    y_hat=forward(x,w)
    dw=gradient(x,y,y_hat)
    w-=alpha*dw
        
    start+=batch_size
    end+=batch_size
    
    
#%%
    #analytical solution
# calculate coefficients using closed-form solution
coeffs = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    











