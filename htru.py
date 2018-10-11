#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:29:10 2018

@author: davitisoselia
"""
import numpy as np
import pandas as pd
from keras.utils  import to_categorical
from sklearn import preprocessing
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
import numpy as np


def index(matrix, a):
    print(matrix.shape)
    return ([(int(a/matrix.shape[1])), a%int(matrix.shape[1])])

my_data = np.array(pd.read_csv('HTRU_2.csv', sep=',',header=None))

x = my_data[:,0:8]
y = my_data[:,8]

y = to_categorical(y)
num_classes = 2
x = preprocessing.scale(x)







model = Sequential()
model.add(Dense(x.shape[1], activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x,y,epochs = 200,verbose=0)

model.evaluate(x, y)

X=x

weights = model.weights # weight tensors
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # sample weights
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]
get_gradients = K.function(inputs=input_tensors, outputs=gradients)

inputs = [X, # X input data
          [1], # sample weights
          [[1]], # y labels
          0 # learning phase in TEST mode
]

#print([a for a in zip(weights, get_gradients(inputs))])
m = [a for a in zip(weights, get_gradients(inputs))]

maxs = []
for i in range(0,len(m),2):
    #min = index(m[i][1], np.abs(np.argmax(m[i][1])))
    max = index(m[i][1], np.abs(np.argmin(m[i][1])))
    maxs.append(max)
w = model.get_weights()
ind  = 0
for i in range(0,len(m),2):
    #print(maxs[i])
    weight_to_change = [i,maxs[ind][0],maxs[ind][1]]
    #print(weight_to_change)
    w[weight_to_change[0]][weight_to_change[1]][weight_to_change[2]] = 0
    ind += 1
model.set_weights(w)
model.evaluate(X,y)






