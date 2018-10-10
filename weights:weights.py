#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:19:26 2018

@author: davitisoselia
"""

import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
import numpy as np












model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
a = np.zeros([20,3])
b = np.zeros([20]).reshape(20)


import pandas as pd
data = pd.read_csv("file.csv")
display(len(data))
  
X = data.iloc[:, 0:4].values #features
y = data.iloc[:, [4]].values #labels
model.fit(X,y,epochs = 200)

w = model.get_weights()
model.summary()
'''
model.evaluate(X,y)
w[0][0][3] = w[0][0][3] +15

model.set_weights(w)
model.evaluate(X,y)

'''




'''

# model is a keras Model
weights = model.weights # weight tensors
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

import keras.backend as K
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

print([a for a in zip(weights, get_gradients(inputs))])
m = [a for a in zip(weights, get_gradients(inputs))]
m[0]
min = index(np.argmax(m[0][1]))
max = index(np.argmin(m[0][1]))
'''

def index(matrix, a):
    print(matrix.shape)
    return ([(int(a/matrix.shape[1])), a%int(matrix.shape[1])])





model.fit(X,y,epochs = 200)



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
    max = index(m[i][1], np.abs(np.argmax(m[i][1])))
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

