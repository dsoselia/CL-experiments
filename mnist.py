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
import numpy as np

from sklearn.utils import class_weight

def index(matrix, a):
    #print(matrix.shape)
    return ([(int(a/matrix.shape[1])), a%int(matrix.shape[1])])

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
x = x.astype('float32')
x_test = x_test.astype('float32')
x = x / 255
x_test = x_test/ 255
y = to_categorical(y_train)
y_test = to_categorical(y_test)



#model = Sequential()

input = Input(shape=(x_train.shape[1]**2,))

x = Dense(x.shape[1], activation='relu')(input)
x = Dense(512, activation='relu')(x)
output = Dense(y.shape[1], activation='softmax')
model = Model(input,output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_ints = [y.argmax() for y in y]
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)
model.fit(x,y,epochs = 20,verbose=1, validation_data = (x_test, y_test))

model.evaluate(x, y)
model.evaluate(x_test, y_test)


X=x

weights = model.weights # weight tensors
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors



input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # sample weights
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

#gradients_1: dictionary{layer: weight gradient list}
#for each trainable layer:
#   gradients_1[layer]=get_gradient(layer,nextlayer)


gradients={}
trainables = [x for x in self.model.layers if x.trainable is True]
trainable_weights = [x.get_weights() for x in trainables]
i_max = len(trainables)


for i in range(i_max-1, -1):
   temp_model=Model(input,trainables[i])
   temp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   temp_layers = [la for la in temp_model.layers if la.trainable is True]
   temp_length = len(temp_layers)
   for j in range(0,temp_length):
      temp_layers[j].set_weights(trainable_weights[j])
   gradients[trainables[i]] = temp_model.optimizers.get_gradients(temp_model.total_loss, trainables[i].weights)
   




  
inputs = [X, # X input data
          [1], # sample weights
          [[1]], # y labels
          0 # learning phase in TEST mode
]

#print([a for a in zip(weights, get_gradients(inputs))])
m = [a for a in zip(weights, get_gradients(inputs))]
annihilated = []
maxs = []
for i in range(0,len(m),2):
    maxs.append([])
    min_num = 0
    while(min_num<m[i][1].shape[0]*m[i][1].shape[1]/4):
        #min = index(m[i][1], np.abs(np.argmax(m[i][1])))
        max_val = index(m[i][1], (np.argmax(np.abs(m[i][1]))))
        if (m[i][1][max_val[0]][max_val[1]] != np.abs(m[i][1]).max()):
            print("error")
            print(m[i][1][max_val[0]][max_val[1]])
            print(np.abs(m[i][1]).max())
        m[i][1][max_val[0]][max_val[1]] = 0
        maxs[-1].append(max_val)
        min_num+=1
        

w = model.get_weights()
ind  = 0
for i in range(0,len(m),2):
    #print(ind)
    #print(maxs[ind])
    for max_value in maxs[ind]:
        weight_to_change = [i,max_value[0],max_value[1]]
        w[weight_to_change[0]][weight_to_change[1]][weight_to_change[2]] = 0
        annihilated.append(weight_to_change)
    ind += 1
model.set_weights(w)
model.evaluate(x_test,y_test)

