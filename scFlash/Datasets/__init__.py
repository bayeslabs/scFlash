from flash.core.data import DataModule
from torch.utils.data import random_split
import scanpy as sc
from scFlash.Datasets.GeneExpression import GeneExpressionDataset
import numpy as np
from torch.utils.data import DataLoader
import os
import platform 

class ScDataset(DataModule):

    def __init__(
                 self, 
                 train_path = None, 
                 val_path = None,
                 test_path = None,
                 predict_path = None,
                 split : dict = {}, 
                 normalize : list = [],
                 batch_size : int = 1,
                 predict_batch_size: int = 1,
                 num_workers: int = None,
                 return_vars: list = [],
                 var_names : str = 'gene_symbols',
                 drop_last : bool = True
                ):
        super(ScDataset, self).__init__(num_workers = num_workers)
        
        self.normalize = normalize
        self.train_path = train_path
        self.val_path = val_path 
        self.test_path = test_path 
        self.predict_path = predict_path
        self.split = split
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        for i in return_vars:
            assert i in ['size_factors'], f'The return variables must must belong to [size_factors] : {i} is not recognized'
        self.return_vars = return_vars
        self.drop = drop_last
        self.var_names = var_names
        # self.train_ds = None
        # self.val_ds = None
        # self.test_ds = None 
        # self.predict_ds = None

        self.prepare_data()
        
        if self.train_path:
            self.model_args = self.get_model_args()

        # if self.train_ds is not None:
        #     self.train_dataloader = self.train_dataloader()
        
        # if self.val_ds is not None:
        #     self.val_dataloader = self.val_dataloader()

        # if self.test_ds is not None:
        #     self.test_dataloader = self.test_dataloader()

        # if self.predict_ds is not None:
        #     self.predict_dataloader = self.predict_dataloader()

    def get_model_args(self):

        adata = self.datasets['train']
        shape = adata.X.toarray().shape
        model_args = {'num_inputs': shape[0], 'input_dim': shape[1], 'batch_size':self.batch_size}
        
        return model_args

    def process(self, i, name):

                if i is not None:
                    adata = sc.read_10x_mtx(
                                        i,  
                                        var_names=self.var_names,                
                                        cache=True
                                        )
                    if self.var_names == 'gene_symbols':
                        adata.var_names_make_unique()
                    
                    adata.size_factors = None

                    if 'size_factors' in self.return_vars:
                            sc.pp.normalize_per_cell(adata)
                            data = adata.copy().X.toarray()
                            size_factor_cell = np.sum(data, axis = 1) / np.median(np.sum(data, axis = 1) )
                            adata.size_factors =  size_factor_cell.reshape((-1,1))

                    if self.normalize is not None: 
                            
                        if 'logtrans_input' in self.normalize:
                            sc.pp.log1p(adata)
                        if 'scale_input' in self.normalize:
                            sc.pp.scale(adata)
                   
                    
                else:
                    adata = None

                self.datasets[name] = adata
        
    def prepare_data(self):

            self.datasets = {}

            for i,name in zip([self.train_path, self.val_path, self.test_path, self.predict_path],['train','val','test','predict']):
                self.process(i, name)
            
    def setup(self, stage = None):
            
            if stage == 'fit' or stage is None:

                if self.datasets['train'] is not None:
                    data = self.datasets['train'].X.toarray()
                    size_factor = np.ascontiguousarray(self.datasets['train'].size_factors, dtype = np.float32) if self.datasets['train'].size_factors is not None else None
                    train_ds = GeneExpressionDataset('train', data, size_factor, self.return_vars)

                    if 'test' in self.split and self.datasets['test'] is None:
                        train_ds, self.test_ds = random_split(train_ds, 
                                                                [int(len(train_ds)*0.8), 
                                                                len(train_ds) - int(len(train_ds) * 0.8)] )
                            
                    if 'val' in self.split and self.datasets['val'] is None:
                        train_ds, self.val_ds = random_split(train_ds, 
                                                                [int(len(train_ds)*0.8), 
                                                                len(train_ds) - int(len(train_ds) * 0.8)] )
                    self.train_ds = train_ds  
                 
            
                if self.datasets['predict'] is not None:
                    
                    data = self.datasets['predict'].X.toarray()
                    size_factor = np.ascontiguousarray(self.datasets['predict'].size_factors, dtype = np.float32) if self.datasets['predict'].size_factors is not None else None
                    self.predict_ds = GeneExpressionDataset('predict', data, size_factor, self.return_vars)
                    
                
                
            if stage == 'val' or stage is None:   
                if self.datasets['val'] is not None:
                        data = self.datasets['val'].X.toarray()
                        size_factor = np.ascontiguousarray(self.datasets['val'].size_factors, dtype = np.float32) if self.datasets['val'].size_factors is not None else None
                        self.val_ds = GeneExpressionDataset('val', data, size_factor, self.return_vars)

            if stage == 'test' or stage is None:
                if self.datasets['test'] is not None:
                        data = self.datasets['test'].X.toarray()
                        size_factor = np.ascontiguousarray(self.datasets['test'].size_factors, dtype = np.float32) if self.datasets['test'].size_factors is not None else None
                        self.test_ds = GeneExpressionDataset('test', data, size_factor, self.return_vars)
            
    def add_predict(self, path):

        self.process(path, 'predict')

    def train_dataloader(self) -> DataLoader:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, 
                drop_last = self.drop        
            )

    def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_ds,
                batch_size=self.predict_batch_size,
                shuffle=False,
                num_workers=self.num_workers, 
                drop_last = self.drop   
            )

    def test_dataloader(self) -> DataLoader:
            return DataLoader(
                self.test_ds,
                batch_size=self.predict_batch_size,
                shuffle=False,
                num_workers=self.num_workers, 
                drop_last = self.drop   
                )
        
    def predict_dataloader(self) -> DataLoader:

            return DataLoader(
                self.predict_ds,
                batch_size=self.predict_batch_size,
                shuffle=False,
                num_workers=self.num_workers, 
                drop_last = self.drop   
                )






                                          



