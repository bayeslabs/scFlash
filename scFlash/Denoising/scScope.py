from flash.core import data
import numpy as np
from typing import List
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from scFlash.utils.modules import AutoEncoder, Linear, scTask
from typing import Any

import numpy as np

class scScope(nn.Module):

    def __init__(
            self,
            input_dim,
            batch_size,
            num_inputs,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            helper_class,
            t: int = 2,
            
            **kwargs
        ):

        super(scScope, self).__init__()

        self.t = t
        self.input_dim = input_dim

        self.autoencoder = AutoEncoder(input_dim, encoder_layers_dim, decoder_layers_dim, latent_layer_out_dim,
                                      activation='relu', weight_initializer='normal', weight_init_params={'std': 0.1},
                                      bias_initializer='zeros', batchnorm=False)
        num_batch = num_inputs//batch_size
        self.batch_effect_layer = nn.Linear(num_batch, self.input_dim, bias=False)
        nn.init.zeros_(self.batch_effect_layer.weight)

        impute_layer1 = Linear(self.input_dim, 64, activation='relu',
                                         weight_init='normal', weight_init_params={'std': 0.1},
                                         bias_init='zeros')

        impute_layer2 = Linear(64, self.input_dim, activation=None,
                                         weight_init='normal', weight_init_params={'std': 0.1},
                                         bias_init='zeros')


        self.imputation_model = nn.Sequential(impute_layer1, impute_layer2)
        self.num_batch = num_batch
        self.helper_class = helper_class
        self.is_predict = False
    
    def loss_fn(self, y_pred, input_d):
        
        output_layer_list, use_mask, batch_effect_removal_layer = y_pred
        input_d_corrected = input_d - batch_effect_removal_layer
        val_mask = torch.sign(input_d_corrected)
        for i in range(len(output_layer_list)):
            out_layer = output_layer_list[i]
            if i == 0:
                loss_value = (torch.norm(torch.mul(val_mask, out_layer - input_d_corrected)))
            else:
                loss_value = loss_value + (torch.norm(torch.mul(val_mask, out_layer - input_d_corrected)))
        return loss_value

    def get_loss(self):

        return {'loss': self.loss_fn}

    def forward(self, X):
       
        if not self.is_predict:
            X, exp_batch_input = X
                
            one_hot = torch.zeros_like(X)[:,:self.num_batch].float()
                
            one_hot[...,exp_batch_input] = 1.0

            batch_effect_removal_layer = self.batch_effect_layer(one_hot)
        else:
            batch_effect_removal_layer = 0

        latent_features_list = []
        output_list = []
        

        for i in range(self.t):
            if i == 0:
                x = F.relu(X - batch_effect_removal_layer)
            else:
                imputed = self.imputation_model(output)
                imputed = torch.mul(1 - torch.sign(X), imputed)
                x = F.relu(imputed + X - batch_effect_removal_layer)
            latent_features, output = self.autoencoder(x)
            output_list.append(output)
            latent_features_list.append(latent_features)

        return output_list, latent_features_list, batch_effect_removal_layer
    
    def step(self, batch: Any, batch_idx: int,) -> Any:
        
        x, y = batch
        batch = (x, batch_idx), y
        output = self.helper_class.step(batch, batch_idx)
        
        return output
    

    
    def _predict(self, trainer, datamodule):

        self.is_predict = True
        o = self.helper_class._predict(trainer, datamodule)
        self.is_predict = False

        return o

    def _func(self, x):
        
        return np.concatenate([i[0][1] for i in x], 0)