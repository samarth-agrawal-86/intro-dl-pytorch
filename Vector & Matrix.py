# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 09:55:00 2018

@author: AGARWAS3
"""

import pandas as pd
import numpy as np

x = np.linspace(0,1, num = 50)
x.shape

In order for model to work we ned to pass matrix
Matrix with 1 column is not same as vector


right now it's a vector. it means it has only 1 axis. X ka rank is 1. Rank of a variable is equal to len of it's shape. ie how many axis it has

len(x.shape)

n dimensional array means - tensor of rank n


# how do we turn 1D array to 2D matrix
x[:,None]

Note how this is different from 
x[None, :]

But things will get complicated as you start working in multiple dimensional tensors with image processing and all
so a good way to write is write like this
x[...,None]
Fill as many dimensions as you need... then add unit axis

