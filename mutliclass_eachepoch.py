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
from keras.layers import Input, Dense, Add
from keras.layers import Dropout
from keras.callbacks import LambdaCallback
from keras import backend as K
from sklearn.utils import class_weight
from keras.datasets import mnist

#load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_lower = [] # corresponding to labels 0-2
y_train_lower = []
x_test_lower = []
y_test_lower = []

x_train_middle = [] #3-5
y_train_middle = []
x_test_middle = []
y_test_middle = []

x_train_upper = [] # 6-9
y_train_upper = []
x_test_upper = []
y_test_upper = []

trawi=False

#divide mnist into lower,upper
for i, label in enumerate(y_train):
    x = x_train[i]
    y = y_train[i]
    if label < 3:
        x_train_lower.append(x)
        y_train_lower.append(y)
    elif label<6:
        x_train_middle.append(x)
        y_train_middle.append(y)
    else:
        x_train_upper.append(x)
        y_train_upper.append(y)


for i, label in enumerate(y_test):
    x = x_test[i]
    y = y_test[i]
    if label < 3:
        x_test_lower.append(x)
        y_test_lower.append(y)
    elif label < 6:
        x_test_middle.append(x)
        y_test_middle.append(y)
    else:
        x_test_upper.append(x)
        y_test_upper.append(y)

x_train_lower = np.array(x_train_lower)
x_train_middle = np.array(x_train_middle)
x_train_upper = np.array(x_train_upper)

x_test_lower = np.array(x_test_lower)
x_test_middle = np.array(x_test_middle)
x_test_upper = np.array(x_test_upper)

y_train_lower = np.array(y_train_lower)
y_train_middle = np.array(y_train_middle)
y_train_upper = np.array(y_train_upper)

y_test_lower = np.array(y_test_lower)
y_test_middle = np.array(y_test_middle)
y_test_upper = np.array(y_test_upper)

training_stage_datasets = [(x_train_lower,y_train_lower), (x_train_middle,y_train_middle), (x_train_upper,y_train_upper)]

#define the model
#input = Input(shape=(x_train.shape[1]**2,))
model_input = Input(shape=(x_train.shape[1]**2,))
x1 = Dense(512, activation='relu')(model_input)
#x1 = Dropout(0.1)(x1)
x1 = Dense(512, activation='relu')(x1)
#x1 = Dropout(0.1)(x1)
output_base_lt = Dense(10)(x1)
lt_output=Activation('softmax')(output_base_lt)



model = Model(model_input,lt_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
iters=20 #FIXME
avgscores = list()
training_stage = 0
weight_proc_epoch=1
def weights_proc(epoch,logs):
    if epoch is weight_proc_epoch:
        global avgscores
        global training_stage
        global trawi
        current_x = training_stage_datasets[training_stage]
        
        
        temp_avg_scores=[]
        
        X = np.array(current_x[0])
        X = X.reshape(X.shape[0], X.shape[1]**2)
        X = X.astype('float32')
        X = X / 255
        Y = to_categorical(np.append(current_x[1], 9))[:-1]
        
        
         #save current weights in initweights
        t_model = Model(model_input,lt_output) #create a t_model
        t_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        initweights = t_model.get_weights()
        t_model.set_weights(initweights) #load the initweights in the t_model
        initloss = t_model.evaluate(X,Y,128)[0] #forward pass t_model and calculate and store loss in initloss
        scores = list() #for each weight array in the weights create a weight_shape * iters array of zeros to store the weight score data (scores)
        for i in range(0,len(initweights)):
            shape = list(initweights[i].shape)
            shape.append(iters)
            scores.append(np.zeros(shape))
            
        for i in range(0,iters): #repeat the following iters times (i)
            #print("weights process iteration " + str(i))
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
            t_model.set_weights(initweights) #reload initweights in t_model
        
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
            temp_avg_scores.append(av_matrix)
        if len(avgscores) is 0:
            avgscores = temp_avg_scores
        else:
            for i in range(0,len(avgscores)):
                avgscores[i] = ((training_stage)*avgscores[i]+temp_avg_scores[i])/(training_stage + 1)
    if len(avgscores)>0:
        print("!!!LOADING FROZEN WEIGHTS!!!")
        if not trawi:
            trawi=model.get_weights()
        saved_w = filtered_weights()
        new_values = overwrite(saved_w)
        model.set_weights(new_values)

def initial_train(epochs=1):
    X = np.array(x_train_lower)
    X = X.reshape(X.shape[0], X.shape[1]**2)
    X = X.astype('float32')
    X = X / 255
    #Y = to_categorical(y_train_lower)[:-1]
    Y = to_categorical(np.append(y_train_lower, 9))[:-1]
    
    Xt = np.array(x_test_lower)
    Xt = Xt.reshape(Xt.shape[0], Xt.shape[1]**2)
    Xt = Xt / 255
    #Yt = to_categorical(y_test_lower)[:-1]
    Yt = to_categorical(np.append(y_test_lower, 9))[:-1]
    
    model.fit(X,Y,epochs = epochs,verbose=1, validation_data = (Xt, Yt),callbacks=[LambdaCallback(on_epoch_end=weights_proc)])

def index(matrix, a):
    #print(matrix.shape)
    return ([(int(a/matrix.shape[1])), a%int(matrix.shape[1])])

def evaluate( x,y):
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1]**2)
    x = x.astype('float32')
    x = x / 255
    y = to_categorical(np.append(y, 9))[:-1]
    #print(y[:5])
    print(model.evaluate(x, y, verbose=1))

def eval_all():
    print("lower")
    (evaluate((x_train_lower), (y_train_lower)))
    (evaluate((x_test_lower), (y_test_lower)))

    print("middle")
    (evaluate((x_train_middle), (y_train_middle)))
    (evaluate((x_test_middle), (y_test_middle)))

    print("upper")
    (evaluate((x_train_upper), (y_train_upper)))
    (evaluate((x_test_upper), (y_test_upper)))

    print("all")
    (evaluate(x_train,y_train))
    (evaluate(x_test,y_test))



def filtered_weights(weights_to_skip=2):
    #percentile = 100 - (training_stage+1)*divisor*100
    percentile = 100 - divisor*100
    m = avgscores
    weights = model.get_weights()
    filtered = []
    for i in range(0, len(weights)-weights_to_skip):
        current_shape = weights[i].shape
        if len(current_shape)>1:
            unrolled_scores = m[i].flatten()
            unrolled_weights = weights[i].flatten()
            minp = np.percentile(unrolled_scores,percentile)
            for j in range(0,len(unrolled_scores)):
                if unrolled_scores[j]<minp:
                    unrolled_weights[j]=0
            filtered.append(unrolled_weights.reshape(current_shape))
        else:
            #print(weights[i].shape)
            filtered.append(weights[i])
    
    for i in range(len(weights)-weights_to_skip, len(weights)):
        filtered.append(weights[i])
    return filtered

j=[]
def overwrite(mats):
    print(len(mats))
    values = model.get_weights()
    new_mats=[]
    new_values=[]
    for matrix in mats:
        new_mats.append((matrix==0).astype(int))
    for i in range(len(values)):
        new_values.append((new_mats[i]*values[i])+mats[i]) 
    return new_values

    
def train_on_set(x , y, x_test, y_test, class_number, epochs=4):  
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
    model.fit(x,y,epochs = epochs,verbose=1, validation_data = (x_test, y_test), callbacks=[LambdaCallback(on_epoch_end=weights_proc)])
    


divisor = 1.0
weight_proc_epoch=3
iters = 3
training_stage = 0
initial_train(4)
eval_all()

training_stage=training_stage + 1
train_on_set(x_train_middle, y_train_middle,x_test_middle, y_test_middle, 10, 4)
eval_all()    

#training_stage=training_stage + 1
#for i in range(1):
#    train_on_set(x_train_upper, y_train_upper,x_test_upper, y_test_upper, 10, 1)
#eval_all()    
