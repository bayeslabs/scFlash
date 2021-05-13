import numpy as np
import pandas as pd
import torch
from torch import distributions
import yaml 
import os 

def nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

def justify(mode, text, n=15):
    if mode == "l":
        return text.ljust(n)
    if mode == "c":
        return text.center(n)
    if mode == "r":
        return text.rjust(n)

def _nelem(x):
    nelem = torch.sum(torch.tensor(~torch.isnan(x), dtype=torch.float32))
    return torch.where((torch.eq(nelem, torch.tensor(0.))), torch.tensor(1.), nelem)


def reduce_mean(x):
    nelem = _nelem(x)
    x = nan2zero(x)
    return torch.sum(x) / nelem

def log_config(config):
    for key, value in config.items():
        key = justify("l", str(key))
        value = justify("c", str(value))
        print(key+":"+value)

def read_yaml(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist.")
    config = yaml.safe_load(open(config_path, "r"))
    
    return config

def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = distributions.normal.Normal(0, 1)

    if sigma == 0:
        x = torch.zeros(size=uniform.shape)
        x = x + mu
        return x

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x


def truncated_normal(mean, stddev, shape):
    uniform = torch.from_numpy(np.random.normal(loc=mean, scale=stddev, size=shape))
    return parameterized_truncated_normal(uniform, mu=mean, sigma=stddev, a=-(2*stddev), b=(2*stddev))