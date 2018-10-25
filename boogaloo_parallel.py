acc_lower =[]
acc_middle =[]
acc_upper =[]

for i in range(3) :   
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
    from keras.layers import Input, Dense, average
    from keras.layers import Dropout
    from keras.callbacks import LambdaCallback
    from keras import backend as K
    from sklearn.utils import class_weight
    from keras.datasets import mnist
    import os
    TRAS = False
     
    
    
    try:
        os.remove('lower.csv')
        os.remove('middle.csv')
        os.remove('upper.csv')
    except:
        pass
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
    
    classes=10
    
    staged_x=[x_train_lower,x_train_middle,x_train_upper]
    staged_y=[y_train_lower,y_train_middle,y_train_upper]
    
    #define the model
    #input = Input(shape=(x_train.shape[1]**2,))
    input_layer = Input(shape=(x_train.shape[1]**2,))
    x = Dense(256, activation='relu')(input_layer)
    x = Dense(512, activation='relu')(x)
    continual_beforemax = Dense(classes)(x)
    continual_out = Activation('softmax')(continual_beforemax)
    
    x2 = Dense(256, activation='relu')(input_layer)
    x2 = Dense(512, activation='relu')(x2)
    x2 = Dense(classes, activation='relu')(x2)
    x2 = average([x2,continual_beforemax])
    output = Activation('softmax')(x2)
    cont_model= Model(input_layer,continual_out)
    model = Model(input_layer,output)
    model = cont_model #KEEP THIS LINE TO DISABLE THE NON-CONTINUOUS MODEL
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cont_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    input_1 = Input(shape=(784,))
    x_1 = Dense(256, activation='relu')(input_1)
    x_1 = Dense(512, activation='relu')(x_1)
    output_1 = Dense(classes, activation='softmax')(x_1)
    model_1 = Model(input_1,output_1)
    model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    
    training_stage = 0
    iters=200
    avgscores = list()
    def weights_proc(epoch,logs):
        global avgscores
        if epoch is 2 or TRAS:
            loc_av=[]
            X = np.array(staged_x[training_stage])
            X = X.reshape(X.shape[0], X.shape[1]**2)
            X = X.astype('float32')
            X = X / 255
            Y = to_categorical(np.append(staged_y[training_stage], classes-1))[:-1]
            
            
            initweights = cont_model.get_weights() #save current weights in initweights
            t_model = Model(model.input,continual_out) #create a t_model
            t_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
                l_difference = abs(initloss - t_model.evaluate(X,Y,128, verbose = 0)[0]) #forward-pass t_model, calculate the loss, store abs(loss-initloss) in l_difference
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
                loc_av.append(av_matrix)
            if len(avgscores) is 0:
                avgscores = loc_av
            else:
                for i in range(0,len(avgscores)):
                    loc_zeros = [avgscores[i]==0]
                    shape = loc_zeros.shape
                    loc_zeros=loc_zeros.flatten()
                    loc_av[i]=loc_av[i].flatten()
                    for j in range(0,len(loc_zeros)):
                        if loc_zeros[j] is 0:
                            loc_av[i][j]=0
                    avgscores[i] = avgscores[i].reshape(shape) 
                    avgscores[i] = ((training_stage)*avgscores[i]+loc_av[i])/(training_stage + 1)
    
    
    def train(model):
        X = np.array(x_train_lower)
        X = X.reshape(X.shape[0], X.shape[1]**2)
        X = X.astype('float32')
        X = X / 255
        #Y = to_categorical(y_train_lower)[:-1]
        Y = to_categorical(np.append(y_train_lower, classes-1))[:-1]
        
        Xt = np.array(x_test_lower)
        Xt = Xt.reshape(Xt.shape[0], Xt.shape[1]**2)
        Xt = Xt.astype('float32')
        Xt = Xt / 255
        #Yt = to_categorical(y_test_lower)[:-1]
        Yt = to_categorical(np.append(y_test_lower, classes-1))[:-1]
        
        model.fit(X,Y,epochs = 4,verbose=0, validation_data = (Xt, Yt),callbacks=[LambdaCallback(on_epoch_end=weights_proc)])
        res = list()
        res.append(model.evaluate(X, Y))
        res.append(model.evaluate(Xt, Yt))
        return res
        
        
    res = train(model)
    
    def index(matrix, a):
        #print(matrix.shape)
        return ([(int(a/matrix.shape[1])), a%int(matrix.shape[1])])
    def evaluate(x,y):
        x = np.array(x)
        x = x.reshape(x.shape[0], x.shape[1]**2)
        x = x.astype('float32')
        x = x / 255
        y = to_categorical(np.append(y, classes-1))[:-1]
        #print(y[:5])
        print(model.evaluate(x, y, verbose=0))
        a_1 = model.evaluate(x, y, verbose=0)
        a_2 = model_1.evaluate(x,y, verbose = 0)
        print(a_1)
        print(a_2)
        return [a_1[0], a_1[1],a_2[0], a_2[1]]
        
    def get_safe_weights(weights_to_skip=2):
        #percentile = 100 - (training_stage+1)*divisor*100
        percentile = 100 - divisor*100
        m = avgscores
        weights = cont_model.get_weights()
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
    
    def overwrite(mats):
        values = cont_model.get_weights()
        new_mats=[]
        new_values=[]
        for matrix in mats:
            new_mats.append((matrix==0).astype(int))
        for i in range(len(values)):
            new_values.append((new_mats[i]*values[i])+mats[i])
        #print(values)
        #print(new_values)
        return new_values
    divisor = 0.8
    save_w  = get_safe_weights()
    
    def log( values , filename = 'lower.csv'):
        file=open(filename,'a+')
        log1 = ""
        for val in values:
            log1 += str(val)+" , "
        log1 += '\n'
        file.write(log1)
        file.close()    
    
    def train1(x , y, x_test, y_test, class_number, model, epochs = 3 ):
        x = np.array(x)
        x_test = np.array(x_test)
        x = x.reshape(x.shape[0], x.shape[1]**2)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
        x = x.astype('float32')
        x_test = x_test.astype('float32')
        x = x / 255
        x_test = x_test/ 255
        y = to_categorical(np.append(y, classes-1))[:-1]
        y_test = to_categorical(np.append(y_test, classes-1))[:-1]
    
        
        y_ints = [y.argmax() for y in y]
        class_weights = class_weight.compute_class_weight('balanced',
                                                         np.unique(y_ints),
                                                         y_ints)
        model.fit(x,y,epochs = epochs,verbose=0, validation_data = (x_test, y_test))
        
        #model.evaluate(x, y)
        #model.evaluate(x_test, y_test)
        return model
    
    def train_emp(x , y, x_test, y_test, class_number, model, epochs = 1 ):
        x = np.array(x)
        x_test = np.array(x_test)
        x = x.reshape(x.shape[0], x.shape[1]**2)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
        x = x.astype('float32')
        x_test = x_test.astype('float32')
        x = x / 255
        x_test = x_test/ 255
        y = to_categorical(np.append(y, classes-1))[:-1]
        y_test = to_categorical(np.append(y_test, classes-1))[:-1]
    
        
        y_ints = [y.argmax() for y in y]
        class_weights = class_weight.compute_class_weight('balanced',
                                                         np.unique(y_ints),
                                                         y_ints)
        model.fit(x,y,epochs = epochs,verbose=0, validation_data = (x_test, y_test))
    
        return model
    
    
    
    
    model_1 = train_emp(x_train_lower, y_train_lower, x_test_lower, y_test_lower, classes, model_1, 3 )
    
    res = (evaluate((x_test_lower), (y_test_lower)))
    log(res)
    
    
    
    training_stage+=1
    for i in range(3):
        if(i == 3 ):
            TRAS = True
        model = train1(x_train_middle, y_train_middle,x_test_middle, y_test_middle, classes, model, 1)
        new_values = overwrite(save_w)
        cont_model.set_weights(new_values)
        model_1 = train_emp(x_train_middle, y_train_middle,x_test_middle, y_test_middle, classes, model_1, 1 )
        res = (evaluate((x_test_lower), (y_test_lower)))
        log(res)
        res = (evaluate((x_test_middle), (y_test_middle)))
        log(res, 'middle.csv')
        
    TRAS = False
    save_w  = get_safe_weights()
    training_stage+=1
    for i in range(3):
        model = train1(x_train_upper, y_train_upper,x_test_upper, y_test_upper, classes, model, 1)
        new_values = overwrite(save_w)
        cont_model.set_weights(new_values)
        model_1 = train_emp(x_train_upper, y_train_upper,x_test_upper, y_test_upper, classes, model_1, 1)
        res = (evaluate((x_test_lower), (y_test_lower)))
        acc_lower.append(res)
        log(res)
        res = (evaluate((x_test_middle), (y_test_middle)))
        acc_middle.append(res)
        log(res, 'middle.csv')
        res = (evaluate((x_test_upper), (y_test_upper)))
        acc_upper.append(res)
        log(res, 'upper.csv')
        
    (evaluate((x_test_lower), (y_test_lower)))
    (evaluate((x_test_middle), (y_test_middle)))
    (evaluate((x_test_upper), (y_test_upper)))
    #(evaluate(x_test,y_test))

acccs = [selected[1] for selected in acc_lower[2::3]] 
print(np.average(acccs))
acccs = [selected[1] for selected in acc_middle[2::3]] 
print(np.average(acccs))
acccs = [selected[1] for selected in acc_upper[2::3]] 
print(np.average(acccs))

acccs = [selected[1] for selected in acc_lower[2::3]] 
print(np.median(acccs))
acccs = [selected[1] for selected in acc_middle[2::3]] 
print(np.median(acccs))
acccs = [selected[1] for selected in acc_upper[2::3]] 
print(np.median(acccs))