import numpy as np
import pandas as pd
import torch
from scFlash.utils.preprocessing import StandardScaler
from scFlash.utils.preprocessing import rank_bar_line as rank_bar
import torch.nn as nn

class PCA(nn.Module):

    def __init__(self, data, max_components: int = 10, q: int = None, center: bool = True, niter: int = 2, scaling: bool = True, **kwargs):

        self.data = data
        self.max_components = max_components
        self.q = q
        self.center = center
        self.niter = niter
        self.scaling = scaling
        self.shape = data.size()

    def scale(self, data):

        scaler = StandardScaler(data)
        return scaler.forward()

    def PCA_rank(self, S):

        self.eigenvalues = torch.mul(S, S)/(self.shape[0]-1)
        self.sum_eigen_values = torch.sum(self.eigenvalues)

        self.PCs = []
        for i in range(1, self.eigenvalues.size()[0]+1):

            self.PCs += ['PC_'+str(i)]

        self.pc_rank = pd.DataFrame(
            {'PC': self.PCs, 'Proportion of Variance': self.eigenvalues.numpy()/self.sum_eigen_values})

        chart = rank_bar(self.pc_rank, 'PCA', 'Proportion of Variance', 'PC')

    def forward(self, plot_rank: bool = True):
        with torch.no_grad():
            if self.scaling:
                self.scale(self.data)

            self.U, self.S, self.V = torch.pca_lowrank(
                self.data, q=self.q, center=self.center, niter=self.niter)
            self.PCA_rank(self.S)
            result = torch.matmul(self.data, self.V[:, :self.max_components])
            return result
