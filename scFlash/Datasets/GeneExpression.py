# The structure of the dataset, posterior and trainer classes are based on the structures used in ScVI
# The implementation of ScVI by YosefLab is available on GitHub

import numpy as np
import pandas as pd
import scipy.sparse
from torch.utils.data import Dataset
import scanpy
from typing import Union, Dict, List
from collections import OrderedDict, defaultdict
import logging



class GeneExpressionDataset(Dataset):

    def __init__(self, split, data, factors, return_vars):

        
        self.split = split
        self.data = np.ascontiguousarray(data, dtype = np.float32)
        self.batch_indices = np.eye(len(self))
        self.size_factors = factors
        self.return_vars = return_vars

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        
        return_vars = np.concatenate([self.__dict__[i][item] for i in (['data'] + self.return_vars)], -1)
        
        
            
        if self.split == 'predict':
            return return_vars

        else:
            
                
            return return_vars, self.data[item]
        
    





