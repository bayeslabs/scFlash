'''
input -> user preference
returns implementation (either DCA or scScope)
'''

from scFlash.Denoising.DCA import DCA  
from scFlash.Denoising.scScope import scScope 
from scFlash.utils.modules import Base
from typing import Type, Union
import torch

class Denoiser(Base):


    modules = {'DCA' : [DCA,['assert "size_factors" in data_args["return_vars"]']],
               'SCSCOPE': [scScope, ['assert data_args["predict_batch_size"] == data_args["batch_size"]']]}
    
    def denoise(self, x = None):        

        return self.pre_predict(x)

  
        






        