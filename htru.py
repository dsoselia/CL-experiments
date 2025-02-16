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

from sklearn.utils import class_weight

def index(matrix, a):
    #print(matrix.shape)
    return ([(int(a/matrix.shape[1])), a%int(matrix.shape[1])])

my_data = np.array(pd.read_csv('HTRU_2.csv', sep=',',header=None))

x = my_data[:,0:8]
y = my_data[:,8]

y = to_categorical(y)
num_classes = 2
x = preprocessing.scale(x)

x_test = []
y_test = []
test_size = 500
flip = 0
for i in range(test_size):
    if y[i][0] == flip:
        x_test.append(x[i])
        y_test.append(y[i])
        flip = 1-flip
x_test = np.array(x_test)
y_test =  np.array(y_test)
#y_test = y_test.reshape(y_test.shape[0], 1)
x  = x[500:]
y = y[500:]
#y = y.reshape(y.shape[0], 1)



model = Sequential()
model.add(Dense(x.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_ints = [y.argmax() for y in y]
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)
model.fit(x,y,epochs = 20,verbose=1, class_weight=class_weights, validation_data = (x_test, y_test))

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
get_gradients = K.function(inputs=input_tensors, outputs=gradients)

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






