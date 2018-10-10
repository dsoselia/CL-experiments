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
model.evaluate(X,y)
w[0][0][1] = w[0][0][1] +15

model.set_weights(w)
model.evaluate(X,y)
