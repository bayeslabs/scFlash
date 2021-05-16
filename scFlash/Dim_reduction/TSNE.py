import numpy as np
import torch
from scFlash.Dim_reduction.PCA import PCA

class TSNE():

    def __init__(self,data, no_dimensions: int=2, initial_dimensions: int=50, perplexity: float=30.0,
                    cuda: bool=True,tol: float=1e-5, iter: int=100,initial_momentum: float=0.5,
                    final_momentum: float = 0.8, eta: int=500, min_gain: float = 0.01, **kwargs):

        self.data = data
        self.cuda = cuda
        self.tol = tol
        self.perplexity = perplexity
        self.no_dimensions = no_dimensions
        self.iter = iter
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.eta = eta
        self.min_gain = min_gain

        if initial_dimensions >= data.shape[1]:
            self.use_pca = False
        else:
            self.use_pca = True
            self.initial_dimensions = initial_dimensions


        if self.cuda:

            print("Using Cuda")
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)

        else:

            torch.set_default_tensor_type(torch.DoubleTensor)

    def H_beta(self,D, beta):

        P = torch.exp(-D.clone() * beta)

        sum_P = torch.sum(P)

        H = torch.log(sum_P) + beta * torch.sum(D * P) / sum_P
        P = P / sum_P

        return H, P

    def data2p(self):

        self.sum_data = torch.sum(self.data_pca*self.data_pca, 1)
        self.D = torch.add(torch.add(-2 * torch.mm(self.data_pca, self.data_pca.t()), self.sum_data).t(), self.sum_data)

        P = torch.zeros(self.n, self.n,dtype=float)
        self.beta = torch.ones(self.n, 1)
        self.logU = torch.log(torch.tensor([self.perplexity]))
        self.n_list = [i for i in range(self.n)]

        for i in range(self.n):

            if i % 500 == 0:

                print("Computing P-values for point %d of %d..." % (i, self.n))

                self.betamin = None
                self.betamax = None

                self.Di = self.D[i, self.n_list[0:i]+self.n_list[i+1:self.n]]

                (self.H, self.this_P) = self.H_beta(self.Di, self.beta[i])

                Hdiff = self.H - self.logU
                tries = 0

                while torch.abs(Hdiff) > self.tol and tries < 50:

                    if Hdiff > 0:

                        self.betamin = self.beta[i].clone()

                        if self.betamax is None:
                            self.beta[i] = self.beta[i] * 2.0
                        else:
                            self.beta[i] = (self.beta[i] + self.betamax) / 2.0

                    else:

                        self.betamax = self.beta[i].clone()

                        if self.betamin is None:
                            self.beta[i] = self.beta[i] / 2.0
                        else:
                            self.beta[i] = (self.beta[i] + self.betamin) / 2.0

                    (self.H, self.this_P) = self.H_beta(self.Di, self.beta[i])

                    Hdiff = self.H - self.logU
                    tries += 1

                P[i, self.n_list[0:i]+self.n_list[i+1:self.n]] = self.this_P


        return P



    def forward(self):

        with torch.no_grad():
            if self.use_pca:
                pca = PCA(self.data,max_components=self.initial_dimensions)
                self.data_pca = pca.forward()
            else:
                self.data_pca = self.data

            (self.n,self.d) = self.data_pca.shape
            self.Y = torch.randn(self.n, self.no_dimensions)
            self.dY = torch.zeros(self.n, self.no_dimensions)
            self.iY = torch.zeros(self.n, self.no_dimensions)
            self.gains = torch.ones(self.n, self.no_dimensions)

            self.P = self.data2p()
            self.P = self.P + self.P.t()
            self.P = self.P / torch.sum(self.P)
            self.P = self.P * 4.0
            print("get P shape", self.P.shape)
            self.P = torch.max(self.P, torch.tensor([1e-21]))

            for i in range(self.iter):

                self.sum_Y = torch.sum(self.Y*self.Y, 1)
                self.num = -2. * torch.mm(self.Y, self.Y.t())
                self.num = 1. / (1. + torch.add(torch.add(self.num, self.sum_Y).t(), self.sum_Y))
                self.num[range(self.n), range(self.n)] = 0.
                self.Q = self.num / torch.sum(self.num)
                self.Q = torch.max(self.Q, torch.tensor([1e-12]))

                PQ = self.P - self.Q
                for j in range(self.n):
                    self.dY[j, :] = torch.sum((PQ[:, j] * self.num[:, j]).repeat(self.no_dimensions, 1).t() * (self.Y[j, :] - self.Y), 0)

                if i < 20:
                    self.momentum = self.initial_momentum
                else:
                    self.momentum = self.final_momentum

                self.gains = (self.gains + 0.2) * ((self.dY > 0.) != (self.iY > 0.)).double() + (self.gains * 0.8) * ((self.dY > 0.) == (self.iY > 0.)).double()
                self.gains[self.gains < self.min_gain] = self.min_gain
                self.iY = self.momentum * self.iY - self.eta * (self.gains * self.dY)
                self.Y = self.Y + self.iY
                self.Y = self.Y - torch.mean(self.Y, 0)

                if (i + 1) % 10 == 0:
                    C = torch.sum(self.P * torch.log(self.P / self.Q))
                    print("Iteration %d: error is %f" % (i + 1, C))


                if i == 50:
                    self.P = self.P / 2.


        return self.Y
