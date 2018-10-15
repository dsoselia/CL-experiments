#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 18:43:53 2018

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
from keras.layers import Dropout

import numpy as np
from keras import backend as K

from sklearn.utils import class_weight

def index(matrix, a):
    #print(matrix.shape)
    return ([(int(a/matrix.shape[1])), a%int(matrix.shape[1])])

from keras.datasets import mnist

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
    
x_train_lower = [] # corresponding to labels 0-4
y_train_lower = []
x_test_lower = []
y_test_lower = []
    
x_train_upper = [] # corresponding to labels 5-9
y_train_upper = []
x_test_upper = []
y_test_upper = []
    
for i, label in enumerate(y_train):
    x = x_train[i] 
    y = y_train[i]
    if label < 5:  
        x_train_lower.append(x)
        y_train_lower.append(y)
    else:                           
        x_train_upper.append(x)
        y_train_upper.append(y)
                
for i, label in enumerate(y_test):
    x = x_test[i] 
    y = y_test[i]
    if label < 5:  
        x_test_lower.append(x)
        y_test_lower.append(y)
    else:                          
        x_test_upper.append(x)
        y_test_upper.append(y)
        
def train(x , y, x_test, y_test, class_number, model, epochs = 3 ):
    x = np.array(x)
    x_test = np.array(x_test)
    x = x.reshape(x.shape[0], x.shape[1]**2)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
    x = x.astype('float32')
    x_test = x_test.astype('float32')
    x = x / 255
    x_test = x_test/ 255
    y = to_categorical(np.append(y, 9))[:-1]
    y_test = to_categorical(np.append(y_test, 9))[:-1]

    
    y_ints = [y.argmax() for y in y]
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_ints),
                                                     y_ints)
    model.fit(x,y,epochs = epochs,verbose=1, validation_data = (x_test, y_test))
    
    model.evaluate(x, y)
    model.evaluate(x_test, y_test)
    return model

def evaluate(x,y):
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1]**2)
    x = x.astype('float32')
    x = x / 255
    y = to_categorical(np.append(y, 9))[:-1]
    #print(y[:5])
    print(model.evaluate(x, y))


devisor = 0.4
def get_safe_weights_caller(x,y, model):
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1]**2)
    x = x.astype('float32')
    x = x / 255
    y = to_categorical(np.append(y, 9))[:-1]
    return get_safe_weights(x,y,model)

def get_safe_weights(x,y,model):
    X=x
    
    weights = model.weights # weight tensors
    gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
    
    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # sample weights
                     model.targets[0], # labels
                     K.learning_phase(), # train or test mode
    ]
    #get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    
    inputs = [X, # X input data
              [1], # sample weights
              y, # y labels
              0 # learning phase in TEST mode
    ]
    
    #print([a for a in zip(weights, get_gradients(inputs))])
    trainables = [x for x in model.layers if x.trainable is True]
    trainable_weights = [x.get_weights() for x in trainables]
    
    def get_gradients():
      gradients=[]
      i_max = len(trainables)
      for i in reversed(range(0, i_max)):
         temp_model=Model(model.layers[0],trainables[i])
         temp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
         temp_layers = [la for la in temp_model.layers if la.trainable is True]
         temp_length = len(temp_layers)
         for j in range(0,temp_length):
            temp_layers[j].set_weights(trainable_weights[j])
         gradients.append(temp_model.optimizers.get_gradients(temp_model.total_loss, trainables[i].weights))
      return gradients
      
    m = [a for a in zip(trainables, get_gradients())]
    annihilated = []
    meta_maxs=p[] 
    for i in range(0,len(m)-1):
       maxs = []
       for j in m[i][1]:
         maxs.append([])
         min_num = 0
         while(min_num<j.shape[0]*mj.shape[1]*devisor):
               max_val = index(j, (np.argmax(np.abs(j))))
               j[max_val[0]][max_val[1]] = 0
               maxs[-1].append(max_val)
               min_num+=1
       meta_maxs.append(maxs)
    
    changed_weights = []
    for i in range(0,len(m)-1):
      l = trainables[i].get_weights()
      maxs=meta_maxs_[i]
      for j in range(0,m[i][1]):
        #print(ind)
        #print(maxs[ind])
        for max_value in maxs[j]:
            weight_to_change = [j,max_value[0],max_value[1]]
            l[weight_to_change[0]][weight_to_change[1]][weight_to_change[2]] = 0
            annihilated.append(weight_to_change)
      changed_weights.append(l)
      
    return changed_weights
#tak = get_safe_weights_caller((x_test_upper), (y_test_upper), model)
'''
model.set_weights(w)
model.evaluate(x_test,y_test)
'''

def overwrite(model,mats):
    values = model.get_weights()
    new_mats=[]
    new_values=[]
    for matrix in mats:
        new_mats.append((matrix==0).astype(int))
    for i in range(len(values)):
        new_values.append((new_mats[i]*values[i])+mats[i])
    #print(values)
    #print(new_values)
    return new_values



model = Sequential()
model.add(Dense(x.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model = train(x_train_lower, y_train_lower,x_test_lower, y_test_lower , 10, model)

safe_weights_stash = get_safe_weights_caller((x_test_lower), (y_test_lower), model)

#print(model.summary())
for i in range(1):
    model = train(x_train_upper, y_train_upper,x_test_upper, y_test_upper ,  10, model, 1)
    new_values = overwrite(model, safe_weights_stash)
    model.set_weights(new_values)

evaluate((x_test_lower), (y_test_lower))
evaluate((x_test_upper), (y_test_upper))
evaluate(x_test,y_test)