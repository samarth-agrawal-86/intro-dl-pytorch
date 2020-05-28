# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:37:40 2018

@author: agarwas3
"""

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

# my model is y = w * x
def forward (x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y-y_pred)**2

def gradient (x, y):
    return 2 * x * (x*w - y)

print ('Predict (before training)' , 4 , forward(4))

alpha = 0.01

for epoch in range (100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient (x_val, y_val)
        w = w - alpha * grad
        print ('\t grad', x_val, y_val, grad)
        l = loss ( x_val, y_val)
        
    print ('Progress: ', epoch, 'w=', w, 'l=', l)
    
print ('Predict (after training)' , 4, round(forward(4)))


#-----
# y = x2 + 2x+1

x_data = [2.0, 3.0, 4.0, 6.0, 8.0]
y_data = [6.0, 15.0, 24.0, 48.0, 80]

w1 = 1.0
w2 = 1.0

# my model is y = ax2  + bx + c
# or y = w1 * x2 + w2 * x + c

def forward (x):
    return w1*x**2 + w2*x 

def loss (x, y):
    y_pred = forward(x)
    return (y - y_pred)**2

def gradient_w1 (x, y):
    y_pred = forward(x)
    return 2 * (y_pred - y) * x**2

def gradient_w2 (x,y):
    y_pred = forward(x)
    return 2* (y_pred-y)*x

print ('Predict (before training)', 8, forward(8))

# weights are optimized as : w = w - alpha * grad
alpha = 0.0001
for epoch in range (500):
    for x_val, y_val in zip(x_data, y_data):
        grad_w2 = gradient_w2 (x_val, y_val)        
        grad_w1 = gradient_w1 (x_val, y_val)
        w1 = w1 - alpha * grad_w1
        w2 = w2 - alpha * grad_w2
        l = loss (x_val, y_val)
        print (x_val, y_val, 'w1 :' ,w1, 'w2 :', w2)
    
    print ('Epoch : ', epoch, 'w1 :' ,w1, 'w2 :', w2, 'l :', l )
    
print ('Predict (after training)' , 10, forward(10))




# AUTO GRAD
'''
We designed the loss function. then we computed our loss function.
then we computed the gradiend of loss function

Manually computing the gradient becomes impossible for say neural network. 
what's better is to use chain rule on Computation Graph
Chain Rule

df/dx = df/dg * dg/dx
'''

import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable (torch.Tensor([1.0]), requires_grad = True)

def forward (x):
    return x* w

def loss (x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


print ('Predict (before training)' , 4 , forward(4))
# forward(4) returns a tensor object. in order to get the value we need to use .data on this
forward(4).data
forward(4).data[0]

alpha = 0.01

for epoch in range (50):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        w.data = w.data - 0.01 * w.grad.data
        # print ('\n grad' , x_val, y_val, w.grad.data)
        
        # Manually reset the gradient to 0
        w.grad.data.zero_()


print ('Predict (after training)' , 4, forward(4).data)

###############################################################################
# LINEAR REGRESSION : PyTorch Way
###############################################################################


# Design your model using class & variables
# Construct loss and optimizer
# Training cycle [forward, backward, update]


import pytorch
from pytorch.autograd import Variable

x_data = Variable (torch.Tensor([ [1.0], [2.0], [3.0] ] ))
y_data = Variable (torch.Tensor([ [2.0], [4.0], [6.0] ]))

w = Variable (torch.Tensor( [1.0] ), requires_grad = True)

# # # Step 1

class Model (torch.nn.Module):
    def __init__ (self):
        super (Model , self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias = True) # one in and one out
        
# we are not going to use our weight here... instead we will rely on the plot we used in init method
    def forward (self, x):
        y_pred = self.linear(x)
        return y_pred
        



# instantiate our Model
model = Model()

# # # Step 2 
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# # # Step 3
for epoch in range (500):
    y_pred = model(x_data)
    print ('Epoch:', epoch, 'Predict', y_pred)
    loss = criterion(y_pred, y_data)
    print ('Loss', loss.data)
    
    optimizer.zero_grad()
    loss.backward()
    print ('Gradient', x_data.grad)
    optimizer.step()
    

# # # Step 4
# Predict output
y_test = Variable(torch.Tensor([4.0]))
print ('Predict (after training)' , model(y_test).data)




###############################################################################
# LOGISTIC REGRESSION : PyTorch Way
###############################################################################

# Step 0 

import torch
from torch.autograd import Variable



x_data = Variable(torch.Tensor ( [[0.0], [2.0], [4.0], [5.0]] ))
y_data = Variable(torch.Tensor ( [[0.0], [0.0], [1.0], [1.0]] ))

w = Variable (torch.Tensor( [1.0] ), requires_grad = True)

# Step 1
class Model (torch.nn.Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)
        
    def forward (self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model ()
# Step 2
criterion = torch.nn.BCELoss (size_average = True) # Binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


# Step 3
for epoch in range (500):
    # Forward
    y_pred = model(x_data)
    
    ''' calculate Loss '''
    loss = criterion(y_pred, y_data)
    
    
    optimizer.zero_grad()
    # Backward
    loss.backward()
    
    # Update
    optimizer.step()
    
    print ('Epoch: ', epoch, 'Predicted value', y_pred, 'Loss : ' , loss, 'Gradient :', w)

y_test1 = Variable ( [[1.0]] )
print ('Selection Yes/No for 1 hr : ', model(y_test1).data>0.5)

y_test2 = Variable ( [[7.0]] )
print ('Selection Yes/No for 7 hr : ', model(y_test2).data>0.5)





###############################################################################
# NEURAL NETWORK : PyTorch Way
###############################################################################

import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

dataset = pd.read_csv('diabetes.csv', header = None, dtype = np.float32)

x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values


x_data = Variable (torch.from_numpy(x))
y_data = Variable (torch.from_numpy(y))

x_data.shape
# torch.Size([759, 8])
y_data.shape
# torch.Size([759])

# Step 1

class Model (torch.nn.Module):
    def __init__ (self):
        super ().__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
    
    
model = Model ()

# Step 2
criterion = torch.nn.BCELoss( size_average = True)
optimizer = torch.optim.SGD( model.parameters(), lr = 0.01)

# Step 3

for epoch in range (500):
    optimizer.zero_grad() 
    
    # forward
    y_pred = model(x_data) # we feed all the data
    
    '''loss calculation'''
    loss = criterion(y_pred, y_data)

    # backward
    loss.backward ()
    
    # update weights
    optimizer.step()
    
    print ('Epoch :', epoch,  'Loss : ', loss, 'Gradient :' )
#    print ('Epoch :', epoch, 'Prediction :', y_pred, 'Loss : ', loss, 'Gradient :' )






###############################################################################
# DATA LOADER
###############################################################################

# we can not need feel all the data in general. 
# we divide all the data into small batches
'''
epoch: cover entire data... let's say 1000 data points 
batch size: number of training examples in one forward & backward pass. lets say 100 our of 1000 data points 
iteration: number of passes it takes to cover entire data set. in this case 10 iterations of 100 batches to cover entire 1000 data points
'''
# This piece of code will be converted into a Custom DataLoader

dataset = pd.read_csv('diabetes.csv', header = None, dtype = np.float32)

x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values


x_data = Variable (torch.from_numpy(x))
y_data = Variable (torch.from_numpy(y))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class DiabetesDataset (Dataset):
    
    def __init__ (self):
        df = pd.read_csv('diabetes.csv', header = None, dtype = np.float32)
        self.len = df.shape[0]
        self.x = df.iloc[:, 0:8].values
        self.y = df.iloc[:,8].values
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        
    def __getitem__ (self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
dataset = DiabetesDataset()

train_loader = DataLoader( dataset, batch_size = 32, shuffle = True, num_workers = 0)

class Model (nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred  = self.sigmoid(self.l3(out2))
        return y_pred
    

model = Model()

criterion = nn.BCELoss( size_average = True)        
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0, dampening=0, weight_decay = 0)

for epoch in range(100):
    for i, data in enumerate (train_loader):
        optimizer.zero_grad()
        inputs, labels = data
        
        inputs, labels = Variable(inputs), Variable(labels)
        
        y_pred = model(inputs)
        
        loss = criterion(y_pred, labels)
        loss.backward()
        
        optimizer.step()
        
        print ('Epoch:', epoch, 'Iteration:', i, 'Loss:', loss)
        





###############################################################################
# DATA LOADER : MNIST
###############################################################################

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


train_dataset = datasets.MNIST(root = './MNISTdata', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = './MNISTdata', train = False, transform = transforms.ToTensor(), download = True)


# Define BatchSize
batch_size = 32
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

for batch_idx, (data, target) in enumerate (train_loader):
    data, target = Variable(data), Variable(target)
    print (data , target)






###############################################################################
# RNN : 
###############################################################################

import torch
from torch import nn, optim


num_classes = 5
input_size, output_size = 5, 5
hidden_size = 5
batch_size = 1
seq_len = 1
num_layers =  1

cell = nn.RN
























