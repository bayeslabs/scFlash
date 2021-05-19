import torch
from torch import nn
from scFlash.utils.modules import AutoEncoder, Linear, scTask
from scFlash.utils.losses import zinb_loss
import numpy as np 

class DCA(nn.Module):
    
    def __init__(
            self,
            input_dim,
            encoder_layers_dim: list = [64],
            decoder_layers_dim: list = [64],
            latent_layer_out_dim: int = 32,
            batchnorm: bool = True,
            activation: str = 'relu',
            weight_initializer='xavier',
            scale_factor = 1.0,
            ridge_lambda = 0.005,
            l1_coeff = 0.,
            l2_coeff = 0.,
            eps = 1e-10,
            **kwargs
    ):
        self.batchnorm = batchnorm
        
        super(DCA, self).__init__()

        self.input_dim = input_dim
        
        self.autoencoder = AutoEncoder(input_dim, encoder_layers_dim = encoder_layers_dim,
                                              decoder_layers_dim=decoder_layers_dim, latent_layer_out_dim=latent_layer_out_dim,
                                              batchnorm=batchnorm, activation=activation,
                                              weight_initializer = weight_initializer, return_output = False)

        self.output_layer_input_dim = self.autoencoder.output_layer_input_dim

        self.output_layer = Linear(self.output_layer_input_dim, self.input_dim,
                                             activation='exp', weight_init = weight_initializer)

        self.theta_layer = Linear(self.output_layer_input_dim, self.input_dim,
                                            activation='softplus', weight_init = weight_initializer)

        self.pi_layer = Linear(self.output_layer_input_dim, self.input_dim,
                                         activation='sigmoid', weight_init = weight_initializer)
        
        self.criterion = zinb_loss

        self.eps = eps
        self.scale_factor = scale_factor
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.ridge_lambda = ridge_lambda
        self._size_factors = None

    def loss_fn(self, y, gt):
        
        loss = self.criterion(gt, y, eps=self.eps, scale_factor=self.scale_factor,
                                             ridge_lambda=self.ridge_lambda)
        return loss

    def reg_fn(self, y, gt):

        l1_reg = torch.tensor(0., device  = torch.device(gt.device))
        l2_reg = torch.tensor(0., device  = torch.device(gt.device))
        for param in self.parameters():
            l1_reg += torch.norm(param, p=1)
            l2_reg += (torch.norm(param, p=2) ** 2)
        loss = self.l1_coeff*l1_reg + self.l2_coeff*l2_reg

        return loss
        
    def get_loss(self):

        return {'zinb_loss': self.loss_fn, 'reg_loss' : self.reg_fn}

    def forward(self, x):
        
        x, size_factor = x[:,:-1], x[:,-1]

        latent_output, decoded = self.autoencoder(x) 
        mean = self.output_layer(decoded)
        
        mean = mean * size_factor.reshape((-1, 1))
        theta = self.theta_layer(decoded)
        pi = self.pi_layer(decoded)
        return mean, theta, pi, latent_output
   
    
    def _func(self, x):
        
        return np.concatenate([i[0] for i in x], 0)
        

    



    