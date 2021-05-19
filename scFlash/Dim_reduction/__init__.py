'''
input -> user preference
returns implementation (either DCA or scScope)
'''

from scFlash.Dim_reduction.PCA import PCA
from scFlash.Dim_reduction.TSNE import TSNE
from scFlash.Dim_reduction.VAE import VAE
from scFlash.utils.modules import Base

from typing import Type, Union
import flash
import torch
import pandas as pd

class Dim_reduction(Base):

    modules = {'PCA' : [PCA,[]] ,'T-sNE': [TSNE,[]],'VAE': [VAE,[]]}

    def reduce(self, x = None):
        if self.model_name == 'VAE':
            return self.pre_predict(x)
        else:
            return self.model(x)



#from PCA import PCA
#from TSNE import TSNE
#from VAE import VAE
#data=pd.read_csv('C:\\Users\\DELL\\Desktop\\Bayes\\data.csv')
#Dim_reduction(algorithm='VAE', data_module=data)
