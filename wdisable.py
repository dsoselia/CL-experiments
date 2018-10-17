import random
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
from keras import backend as K
from sklearn.utils import class_weight
from keras.datasets import mnist

#load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_lower = [] # corresponding to labels 0-4
y_train_lower = []
x_test_lower = []
y_test_lower = []
x_train_upper = [] # corresponding to labels 5-9
y_train_upper = []
x_test_upper = []
y_test_upper = []


#divide mnist into lower,upper
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

x_train_lower = np.array(x_train_lower)
x_train_upper = np.array(x_train_upper)
x_test_lower = np.array(x_test_lower)
x_test_upper = np.array(x_test_upper)

y_train_lower = np.array(y_train_lower)
y_train_upper = np.array(y_train_upper)
y_test_lower = np.array(y_test_lower)
y_test_upper = np.array(y_test_upper)


#define the model
#input = Input(shape=(x_train.shape[1]**2,))
input = Input(shape=(x_train.shape[1]**2,))
x = Dense(256, activation='relu')(input)
x = Dense(512, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(input,output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

iters=2
avgscores = list()
def weights_proc(epoch,logs):
    if epoch is 3:
        X = np.array(x_train_lower)
        X = X.reshape(X.shape[0], X.shape[1]**2)
        X = X.astype('float32')
        X = X / 255
        Y = to_categorical(np.append(y_train_lower, 9))[:-1]
        
        
        initweights = model.get_weights() #save current weights in initweights
        t_model = Model(model.input,model.output) #create a t_model
        t_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        t_model.set_weights(initweights) #load the initweights in the t_model
        initloss = t_model.evaluate(X,Y,128)[0] #forward pass t_model and calculate and store loss in initloss
        scores = list() #for each weight array in the weights create a weight_shape * iters array of zeros to store the weight score data (scores)
        for i in range(0,len(initweights)):
            shape = list(initweights[i].shape)
            shape.append(iters)
            scores.append(np.zeros(shape))
            
        for i in range(0,iters): #repeat the following iters times (i)
            print("weights process iteration " + str(i))
            randomzeros = list() #for each weight tensor in the weights, create a numpy array of the same shape that is 1s everywhere except a randomly picked 20% positions where it's zero. (randomzeros)
            for j in range(0,len(initweights)):
                to_append = np.ones(initweights[j].shape)
                shape = to_append.shape
                to_append = to_append.flatten()
                for element in range(0,len(to_append)):
                    if random.random() < 0.8:
                        to_append[element]=to_append[element]
                    else:
                        to_append[element]=0
                to_append = to_append.reshape(shape)
                randomzeros.append(to_append)
            temp_weights = list() #for each weight array in the weights list make a copy that is element-wise multiplied by the respective matrix from the step above. put those in a list and pass it to t_model.
            for j in range(0,len(initweights)):
                temp_weights.append(randomzeros[j]*initweights[j])
            
            t_model.set_weights(temp_weights)
            l_difference = abs(initloss - t_model.evaluate(X,Y,128)[0]) #forward-pass t_model, calculate the loss, store abs(loss-initloss) in l_difference
            for j in range(0,len(randomzeros)): #for each array in randomzeros (indexed with j), search for zeros, and for each zero at a position x,y store l_difference in scores[j][x][y][i]
                shape = scores[j].shape
                temp_rand_zeros = randomzeros[j].flatten()
                temp_scores = scores[j].reshape(-1,iters)
                for k in range(0,len(temp_rand_zeros)):
                    if temp_rand_zeros[k]<1:
                        temp_scores[k][i]=l_difference
                scores[j] = temp_scores.reshape(shape)
            model.set_weights(initweights) #reload initweights in t_model
        
        for i in range(0,len(scores)):#for each array in scores, get each iters-element column and store the sum (sum) and the number of non-zero elements, then store sum/nonzero-s in a new array of the same dims as the respective weight array (avg_scores).
            av_matrix = np.zeros(initweights[i].shape)
            av_shape=av_matrix.shape
            av_matrix = av_matrix.flatten()
            temp_sc = scores[i].reshape(-1,iters)
            for j in range(0,temp_sc.shape[0]):
                sum = np.sum(temp_sc[j])
                nonzero = np.sum((temp_sc[j] != 0))
                if nonzero>0.000000001:
                    av_matrix[j]=sum/nonzero
            av_matrix = av_matrix.reshape(av_shape)
            avgscores.append(av_matrix)
        for i in range(0,len(avgscores)):
            print(np.where(avgscores[i].flatten()>0))


def train(model):
    X = np.array(x_train_lower)
    X = X.reshape(X.shape[0], X.shape[1]**2)
    X = X.astype('float32')
    X = X / 255
    #Y = to_categorical(y_train_lower)[:-1]
    Y = to_categorical(np.append(y_train_lower, 9))[:-1]
    
    Xt = np.array(x_test_lower)
    Xt = Xt.reshape(Xt.shape[0], Xt.shape[1]**2)
    Xt = Xt.astype('float32')
    Xt = Xt / 255
    #Yt = to_categorical(y_test_lower)[:-1]
    Yt = to_categorical(np.append(y_test_lower, 9))[:-1]
    
    model.fit(X,Y,epochs = 4,verbose=1, validation_data = (Xt, Yt),callbacks=[LambdaCallback(on_epoch_end=weights_proc)])
    res = list()
    res.append(model.evaluate(X, Y))
    res.append(model.evaluate(Xt, Yt))
    return res
    
    
res = train(model)

def evaluate( x,y):
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1]**2)
    x = x.astype('float32')
    x = x / 255
    y = to_categorical(np.append(y, 9))[:-1]
    #print(y[:5])
    print(model.evaluate(x, y, verbose=0))
    
def get_safe_weights(model):
    
    #print([a for a in zip(weights, get_gradients(inputs))])
    m = avgscores
    annihilated = []
    maxs = []
    for i in range(0,len(m)-1,2):
        maxs.append([])
        min_num = 0
        while(min_num<m[i][1].shape[0]*m[i].shape[1]*devisor):
            max_val = index(m[i], (np.argmax(np.abs(m[i]))))
            m[i][max_val[0]][max_val[1]] = 0
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
    return [np.copy(layer) for layer in  w]

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
devisor = 0.2
save_w  = get_safe_weights(model)


def train1(x , y, x_test, y_test, class_number, model, epochs = 3 ):
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
    model.fit(x,y,epochs = epochs,verbose=0, validation_data = (x_test, y_test))
    
    #model.evaluate(x, y)
    #model.evaluate(x_test, y_test)
    return model

for i in range(3):
    model = train1(x_train_upper, y_train_upper,x_test_upper, y_test_upper ,  10, model, 1)
    new_values = overwrite(model, save_w)
    model.set_weights(new_values)
(evaluate((x_test_lower), (y_test_lower)))
(evaluate((x_test_upper), (y_test_upper)))
(evaluate(x_test,y_test))

