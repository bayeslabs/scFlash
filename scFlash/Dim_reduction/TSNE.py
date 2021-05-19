import numpy as np
import torch
from sklearn.manifold import TSNE


class tSNE():

    def __init__(self, data, n_components=10, *, perplexity=30.0, early_exaggeration=12.0,
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
        self.forward()

    def forward(self):

        self.reducer = TSNE(self.data,
                            self.perplexity,
                            self.n_components,
                            self.early_exaggeration,
                            self.n_iter,
                            self.learning_rate,
                            self.n_iter_without_progress,
                            self.min_grad_norm,
                            self.metric,
                            self.init,
                            self.verbose,
                            self.random_state,
                            self.method,
                            self.angle,
                            self.n_jobs,
                            self.square_distances)

        self.Y = self.reducer.fit_transform(self.data)

        return self.Y
