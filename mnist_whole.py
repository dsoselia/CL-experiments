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
        
def train(x , y, x_test, y_test, class_number, model ):
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
    model.fit(x,y,epochs = 3,verbose=1, validation_data = (x_test, y_test))
    
    model.evaluate(x, y)
    model.evaluate(x_test, y_test)
    return model

def evaluate(x,y):
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1]**2)
    x = x.astype('float32')
    x = x / 255
    y = to_categorical(np.append(y, 9))[:-1]
    print(y[:5])
    print(model.evaluate(x, y))


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
model = train(x_train_upper, y_train_upper,x_test_upper, y_test_upper ,  10, model)


evaluate((x_test_upper), (y_test_upper))

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
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    
    inputs = [X, # X input data
              [1], # sample weights
              y, # y labels
              0 # learning phase in TEST mode
    ]
    
    #print([a for a in zip(weights, get_gradients(inputs))])
    m = [a for a in zip(weights, get_gradients(inputs))]
    annihilated = []
    maxs = []
    for i in range(0,len(m)-1,2):
        maxs.append([])
        min_num = 0
        while(min_num<m[i][1].shape[0]*m[i][1].shape[1]/2):
            max_val = index(m[i][1], (np.argmax(np.abs(m[i][1]))))
            m[i][1][max_val[0]][max_val[1]] = 0
            maxs[-1].append(max_val)
            min_num+=1
            
    
    w = model.get_weights()
    ind  = 0
    for i in range(0,len(m)-1,2):
        #print(ind)
        #print(maxs[ind])
        for max_value in maxs[ind]:
            weight_to_change = [i,max_value[0],max_value[1]]
            w[weight_to_change[0]][weight_to_change[1]][weight_to_change[2]] = 0
            annihilated.append(weight_to_change)
        ind += 1
    return w
tak = get_safe_weights_caller((x_test_upper), (y_test_upper), model)
'''
model.set_weights(w)
model.evaluate(x_test,y_test)
'''
