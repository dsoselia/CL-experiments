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
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.callbacks import LambdaCallback

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


saved_weights=[]   
 
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
    model.fit(x,y,epochs = epochs,verbose=1, validation_data = (x_test, y_test),callbacks=[LambdaCallback(on_epoch_end=get_safe_weights)])
    
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

def get_safe_weights(epoch,logs):
   print("hohoho"+str(epoch))
   if epoch is 2:
      #get_gradients = K.function(inputs=input_tensors, outputs=gradients)
      trainables = [x for x in model.layers if x.trainable is True]
      trainable_weights = [x.get_weights() for x in trainables]
      
      def get_gradients():
         gradients=[]
         i_max = len(trainables)
         for i in reversed(range(0, i_max)):
            temp_model=Model(model.layers[0].input,trainables[i].output)
            temp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            temp_layers = [la for la in temp_model.layers if la.trainable is True]
            temp_length = len(temp_layers)
            for j in range(0,temp_length):
               temp_layers[j].set_weights(trainable_weights[j])
            gradients.append(temp_model.optimizer.get_gradients(temp_model.total_loss, trainables[i].weights))
         ret = []
         for elem in gradients:
            retj=[]
            for elemj in elem:
               retj.append(elemj.eval())
            ret.append(retj)
         return ret
         
      m = [a for a in zip(trainables, get_gradients())]
      annihilated = []
      meta_maxs=[] 
      for i in range(0,len(m)-1):
         maxs = []
         for j in m[i][1]:
            maxs.append([])
            min_num = 0
            print(j)
            while(min_num<j.shape[0]*j.shape[1]*devisor):
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
            for max_value in maxs[j]:
               weight_to_change = [j,max_value[0],max_value[1]]
               l[weight_to_change[0]][weight_to_change[1]][weight_to_change[2]] = 0
               annihilated.append(weight_to_change)
         changed_weights.append(l)
      
      saved_weights.append(changed_weights)
      

#tak = get_safe_weights_caller((x_test_upper), (y_test_upper), model)
'''
model.set_weights(w)
model.evaluate(x_test,y_test)
'''

def overwrite(layer,mats):
    values = layer.get_weights()
    new_mats=[]
    new_values=[]
    for matrix in mats:
        new_mats.append((matrix==0).astype(int))
    for i in range(len(values)):
        new_values.append((new_mats[i]*values[i])+mats[i])
    #print(values)
    #print(new_values)
    return new_values



input = Input(shape=(x_train.shape[1]**2,))
x = Dense(x.shape[1], activation='relu')(input)
x = Dense(512, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(input,output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model = train(x_train_lower, y_train_lower,x_test_lower, y_test_lower , 10, model)


#safe_weights_stash = get_safe_weights_caller((x_test_lower), (y_test_lower), model)

#print(model.summary())
for i in range(1):
    model = train(x_train_upper, y_train_upper,x_test_upper, y_test_upper ,  10, model, 1)
    new_values = overwrite(model, saved_weights[0])
    model.set_weights(new_values)

evaluate((x_test_lower), (y_test_lower))
evaluate((x_test_upper), (y_test_upper))
evaluate(x_test,y_test)