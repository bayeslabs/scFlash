# -*- coding: utf-8 -*-
"""VAE_clean.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16Kf3Pd8TFoklgQsq28iip3CNwQNWSIG7
"""

import warnings
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import init
from scFlash.utils.modules import scTask


class Basic():

    def __init__(self):
      pass

    def initialize_weights(self,m):  #add this to utils file
        if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d)):
            #init.uniform_(m.weight.data,a=0.0,b=1.0)
            init.xavier_uniform_(m.weight.data)


    def Layer(self,i, o, activation=None, p=0., bias=True):

        ll=nn.Linear(i, o, bias=bias)
        self.initialize_weights(ll)
        activation = activation.upper() if activation is not None else activation
        model = [ll]
        if activation == 'SELU':
            model += [nn.SELU(inplace=True)]
        elif activation == 'RELU':
            model += [nn.ReLU(inplace=True)]
        elif activation == 'LeakyReLU'.upper():
            model += [nn.LeakyReLU(inplace=True)]
        elif activation == 'Sigmoid'.upper():
            model += [nn.Sigmoid()]
        elif activation == 'Tanh'.upper():
            model += [nn.Tanh()]
        elif activation == 'Swish'.upper():
            model += [Swish()]
        elif type(activation) is str:
            raise ValueError('{} activation not implemented.'.format(activation))

        if p > 0.:
            model += [nn.Dropout(p)]
        return nn.Sequential(*model)



class Encoder(nn.Module):
    """Encoder network for dimensionality reduction to latent space"""
    def __init__(self, input_size, output_size=1, hidden_layer_depth=5,hidden_size=1024, activation='Sigmoid', dropout_rate=0.):
        super(Encoder, self).__init__()
        basic=Basic()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.activation=activation
        self.input_layer = basic.Layer(input_size, hidden_size,activation=activation,p=dropout_rate)
        net = [basic.Layer(hidden_size, hidden_size, activation='RELU',p=dropout_rate) for _ in range(hidden_layer_depth-1)]
        net.append(basic.Layer(hidden_size, hidden_size, activation='Sigmoid',p=dropout_rate))
        self.hidden_network = nn.Sequential(*net)
        self.output_layer = basic.Layer(hidden_size, output_size,activation='Sigmoid')

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_network(out)
        out = self.output_layer(out)
        return out

class Lambda(nn.Module):
    """Application of Gaussian noise to the latent space"""
    def __init__(self, i=1, o=1, scale=1E-3):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps


class Decoder(nn.Module):
    """Decoder network for reconstruction from latent space"""
    def __init__(self, output_size, input_size=1, hidden_layer_depth=5,
                 hidden_size=1024, activation='Sigmoid', dropout_rate=0.):
        super(Decoder, self).__init__()
        basic=Basic()
        self.input_layer = basic.Layer(input_size, input_size, activation='RELU')
        net = [basic.Layer(input_size, hidden_size,activation='RELU', p=dropout_rate)]
        net += [basic.Layer(hidden_size, hidden_size, activation='RELU',p=dropout_rate) for _ in range(hidden_layer_depth-1)]
        net += [basic.Layer(hidden_size, hidden_size, activation='Sigmoid',p=dropout_rate) ]

        self.hidden_network = nn.Sequential(*net)
        self.output_layer = basic.Layer(hidden_size, output_size,activation='Sigmoid')

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_network(out)
        out = self.output_layer(out)
        return out



"""VAE DEFINITION"""

class VAE(nn.Module):
    def __init__(self, input_dim, encoder_size=1, batch_size=100,
                 hidden_layer_depth=3, hidden_size=2048, scale=1E-3,
                 dropout_rate=0.,
                 activation='Sigmoid',
                 verbose=True,
                 **kwargs):

        super(VAE, self).__init__()
        basic = Basic()
        input_size = input_dim
        self.encoder = Encoder(input_size, output_size=encoder_size,
                               hidden_layer_depth=hidden_layer_depth,
                               hidden_size=hidden_size, activation=activation,dropout_rate=dropout_rate)
        self.lmbd = Lambda(encoder_size, encoder_size, scale=scale)
        self.decoder = Decoder(input_size, input_size=encoder_size,
                               hidden_layer_depth=hidden_layer_depth,
                               hidden_size=hidden_size, activation=activation,
                               dropout_rate=dropout_rate)

        self.verbose = verbose

        self.input_size = input_size
        self.encoder_size = encoder_size
        self.apply(basic.initialize_weights)

        """self.lr = lr
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError('Not a recognized optimizer')"""

        #self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.is_fit = False


    def vae_loss(self, x_decoded_mean, x):

        x_decoded_mean = x_decoded_mean[0]
        kl_loss=nn.functional.kl_div(x, x_decoded_mean, size_average=None, reduce=None, reduction='batchmean', log_target=False)
        loss = nn.functional.binary_cross_entropy(nn.functional.softmax(x_decoded_mean,-1), nn.functional.softmax(x,-1), reduction='sum')   # LOSS FUNCTION REQUIRES INPUT IN THE RANGE OF 0 TO 1

        return loss+kl_loss


    def get_loss(self):

        return {'vae_loss': self.vae_loss}

    def forward(self, x):
        u = self.encoder(x)
        u_p = self.lmbd(u)
        out = self.decoder(u_p)
        return out , u
    
    def _func(self, x):
 
        return np.concatenate([i[0] for i in x])



    
