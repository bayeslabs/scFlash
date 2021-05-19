import numpy as np
import torch
from sklearn.manifold import TSNE
import torch.nn as nn

class tSNE(nn.Module):

    def __init__(self, data, n_components=2, *, perplexity=30.0, early_exaggeration=12.0,
                 learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07,
                 metric='euclidean', init='random', verbose=0, random_state=None,
                 method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy'):

        self.data = data.numpy()
        self.perplexity = perplexity
        self.n_components = n_components
        self.early_exaggeration = early_exaggeration
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.square_distances = square_distances

    def forward(self):

        self.reducer = TSNE(perplexity=self.perplexity,
                            n_components=self.n_components,
                            early_exaggeration=self.early_exaggeration,
                            n_iter=self.n_iter,
                            learning_rate=self.learning_rate,
                            n_iter_without_progress=self.n_iter_without_progress,
                            min_grad_norm=self.min_grad_norm,
                            metric=self.metric,
                            init=self.init,
                            verbose=self.verbose,
                            random_state=self.random_state,
                            method=self.method,
                            angle=self.angle,
                            n_jobs=self.n_jobs,
                            square_distances=self.square_distances)

        self.Y = self.reducer.fit_transform(self.data)

        return self.Y

                  