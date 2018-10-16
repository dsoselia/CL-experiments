import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.dense_82 = self.__dense(name = 'dense_82', in_features = 784, out_features = 28, bias = True)
        self.dense_83 = self.__dense(name = 'dense_83', in_features = 28, out_features = 512, bias = True)
        self.dense_84 = self.__dense(name = 'dense_84', in_features = 512, out_features = 256, bias = True)
        self.dense_85 = self.__dense(name = 'dense_85', in_features = 256, out_features = 64, bias = True)
        self.dense_86 = self.__dense(name = 'dense_86', in_features = 64, out_features = 10, bias = True)

    def forward(self, x):
        dense_82        = self.dense_82(x)
        dense_82_activation = F.relu(dense_82)
        dense_83        = self.dense_83(dense_82_activation)
        dense_83_activation = F.relu(dense_83)
        dropout_4       = F.dropout(input = dense_83_activation, p = 0.20000000298023224, training = self.training, inplace = True)
        dense_84        = self.dense_84(dropout_4)
        dense_84_activation = F.relu(dense_84)
        dropout_5       = F.dropout(input = dense_84_activation, p = 0.20000000298023224, training = self.training, inplace = True)
        dense_85        = self.dense_85(dropout_5)
        dense_85_activation = F.relu(dense_85)
        dropout_6       = F.dropout(input = dense_85_activation, p = 0.20000000298023224, training = self.training, inplace = True)
        dense_86        = self.dense_86(dropout_6)
        dense_86_activation = F.softmax(dense_86)
        return dense_86_activation


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
