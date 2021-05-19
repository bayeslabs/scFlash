import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import torch
from torch import nn
from abc import ABC, abstractmethod
import flash 
import torch
from pathlib import Path
from .utils import read_yaml
from scFlash.Datasets import ScDataset as DataSet
from flash import Task 
import torchmetrics
import sys

class ExponentialActivation(nn.Module):

    def __init__(self):
        super(ExponentialActivation, self).__init__()

    def forward(self, x):
        x = torch.exp(x)
        return x

class Linear(nn.Module):

    activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(),
                   'exp': ExponentialActivation(), 'softplus': nn.Softplus()}
    initializers = {'xavier': nn.init.xavier_uniform_, 'zeros': nn.init.zeros_, 'normal': nn.init.normal_}

    def __init__(
            self,
            input_dim,
            out_dim,
            batchnorm = False,
            activation = 'relu',
            dropout = 0.,
            weight_init = None,
            weight_init_params: Dict = None,
            bias_init = None,
            bias_init_params: Dict = None
    ):
        super(Linear, self).__init__()

        if weight_init_params is None:
            weight_init_params = {}
        if bias_init_params is None:
            bias_init_params = {}
        self.batchnorm_layer = None
        self.act_layer = None
        self.dropout_layer = None
        
        self.linear = nn.Linear(input_dim, out_dim)
        
        if weight_init is not None:
            self.initializers[weight_init](self.linear.weight, **weight_init_params)
        if bias_init is not None:
            self.initializers[bias_init](self.linear.bias, **bias_init_params)

        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(out_dim)
        if activation:
            self.act_layer = self.activations[activation]
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm_layer:
            x = self.batchnorm_layer(x)
        if self.act_layer:
            x = self.act_layer(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(
            self,
            input_d,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            batchnorm: bool = True,
            activation: str = 'relu',
            dropout: float = 0.,
            weight_initializer=None,
            weight_init_params: Dict = None,
            bias_initializer = None,
            bias_init_params: Dict = None,
            return_output = True,
            **kwargs
    ):
        super(AutoEncoder, self).__init__()
        if weight_init_params is None:
            weight_init_params = {}
        if bias_init_params is None:
            bias_init_params = {}

        self.input_dim = input_d
        self.batchnorm = batchnorm
        encode_layers = []
        if len(encoder_layers_dim) > 0:
            for i in range(len(encoder_layers_dim)):
                if i == 0:
                    encode_layers.append(Linear(self.input_dim, encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    encode_layers.append(Linear(encoder_layers_dim[i - 1], encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            self.latent_layer_input_dim = encoder_layers_dim[-1]
        else:
            self.latent_layer_input_dim = self.input_dim
        self.encode = nn.Sequential(*encode_layers)

        self.latent_layer = Linear(self.latent_layer_input_dim, latent_layer_out_dim,
                                             batchnorm=self.batchnorm, activation=activation,
                                             dropout=dropout,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)

        decode_layers = []
        if len(decoder_layers_dim) > 0:
            for i in range(len(decoder_layers_dim)):
                if i == 0:
                    decode_layers.append(Linear(latent_layer_out_dim, decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    decode_layers.append(Linear(decoder_layers_dim[i - 1], decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            self.output_layer_input_dim = decoder_layers_dim[-1]
        else:
            self.output_layer_input_dim = latent_layer_out_dim
        self.decode = nn.Sequential(*decode_layers)
        self.return_output = return_output
        if return_output:
            self.output_layer = Linear(self.output_layer_input_dim, self.input_dim,
                                             activation=activation,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)
                

    def forward(self, x):
        encoded = self.encode(x)
        latent = self.latent_layer(encoded)
        output = self.decode(latent)
        if self.return_output:
            output = self.output_layer(output)

        return latent, output

class scTask(Task):

    def __init__(self, learning_rate: float = 1e-3, **kwargs):

        args = ['optimizer',
                'metrics',
               ]
        
        super(scTask, self).__init__(learning_rate=float(learning_rate),
                                    **{i:kwargs[i] for i in kwargs if i in args}) 

    def _train(self, trainer, datamodule):

        trainer.fit(self, datamodule = datamodule)
    
    def _predict(self, trainer, datamodule):

        o = trainer.predict(self, datamodule = datamodule)

        return o

    def _func(self, x):
        
        return self.model._func(x)

    def step(self, batch: Any, batch_idx: int) -> Any:
    
        x, y = batch
        y_hat = self(x)
        output = {"y_hat": y_hat}
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, torchmetrics.metric.Metric):
                metric(y_hat, y)
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                logs[name] = metric(y_hat, y)
        logs.update(losses)
        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())

        output["loss"] = list(losses.values())[0]
        output["logs"] = logs
        output["y"] = y
        return output

class Base(scTask):

    modules = None

    def __init__(
                 self, 
                 config_path : Path = None,
                 model_name: str = None, 
                 model_args : dict = {}, 
                 data_directories : dict = {}, 
                 trainer_args : dict = {}, 
                 optim : dict = {}, 
                 train: bool = False
                ):
        
        if config_path is not None:

            config = read_yaml(config_path)
            model_name = config['model_name'] 
            model_args = config['model_args'] 
            data_args = config['data_args'] 
            trainer_args = config['trainer_args'] 
            optim = config['optim'] 
            train = config['train']
        self.model_name = model_name
        model, conditions = self.modules[model_name]
        try:
            for i in conditions:
                exec(i)

        except Exception as e:
            
            print(f'\nThe condition --|| {i} ||-- Failed! \n\nPlease provide the required parameters in the "{config_path}" file\n' )
            sys.exit()

        print('\nAll conditions checked successfully!\n\nLoading Data\n')

        self.data = DataSet(**data_args)
        data_model_args = self.data.model_args
        super().__init__(**optim)
        self.model = model(helper_class = super(), **data_model_args, **model_args)
        self.loss_fn = self.model.get_loss()
        gpus = -1 if torch.cuda.is_available() else None
        self.trainer = flash.Trainer(**trainer_args, gpus = gpus)

        for i in ['step','_predict', '_train', 'predict', 'train']:

            fnc = getattr(self.model , i, False)
            if fnc:
                self.__dict__[i] = fnc
        
            
        if train:
            
            self.train_model()
            
    
    def train_model(self):

        print('\nTraining the model\n')
        print('\nThe Model Parameters')
        print('---------------------\n---------------------')
        self._train(self.trainer, datamodule = self.data) 
        print(f'\n{self.model_name} is trained and ready to be used!\n')

    def pre_predict(self, x):

        if type(x) == str :

            print('\nNew path detected!\n\nAdding it to the Dataloader\n')
            
            self.data.add_predict(x)
            print('\nData Ready!\n')
      
        else: data = self.data
        print('\nPredicting now\n')
        
        x = self._predict(self.trainer, data)

        return self.model._func(x)
        
