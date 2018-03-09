"""
Project:    GAN_primal_dual
File:       data.py
Created by: louise
On:         08/03/18
At:         4:18 PM
"""
import torch
import torch.nn as nn
import math
from scipy.stats import multivariate_normal
import numpy as np
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def exp1(num_data=1000, sigma=1, meanx=8, dim=2, plot=True):
    if num_data % 2 != 0:
        raise ValueError('num_data should be even. num_data = {}'.format(num_data))

    center = meanx
    # create point position
    d1x = torch.FloatTensor(num_data/2, 1)
    d1y = torch.FloatTensor(num_data/2, 1)
    d1x.normal_(center, sigma * 3)
    d1y.normal_(center, sigma * 1)

    d2x = torch.FloatTensor(num_data/2, 1)
    d2y = torch.FloatTensor(num_data/2, 1)
    d2x.normal_(-center, sigma * 1)
    d2y.normal_(center,  sigma * 3)

    d1 = torch.cat((d1x, d1y), 1)
    d2 = torch.cat((d2x, d2y), 1)

    d = torch.cat((d1, d2), 0)

    # label data points
    label = torch.IntTensor(num_data).zero_()
    for i in range(2):
        label[i*(num_data/2):(i+1)*(num_data/2)] = i

    # Create pdf for log likelihood
    rv1 = multivariate_normal([center,  center], [[math.pow(sigma * 3, 2), 0.0], [0.0, math.pow(sigma * 1, 2)]])
    rv2 = multivariate_normal([-center, center], [[math.pow(sigma * 1, 2), 0.0], [0.0, math.pow(sigma * 3, 2)]])

    if plot:
        plt.figure()
        plt.scatter(d1x.numpy(), d1y.numpy(), label="Gaussian 1")
        plt.scatter(d2x.numpy(), d2y.numpy(), label="Gaussian 2")
        plt.legend()
        plt.show()

    def pdf(x):
        prob = 0.5 * rv1.pdf(x) + 0.5 * rv2.pdf(x)
        return prob

    def sumloglikelihood(x):
        return np.sum(np.log((pdf(x) + 1e-10)))

    return d, label, sumloglikelihood


def get_distribution_sampler(mu, sigma):
    """
    Generate target data
    :param mu: float
    :param sigma: float
    :return:
    """
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian


def get_generator_input_sampler():
    """
    Generate input data for Generator
    :return:
    """
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian


class DataDistribution(object):
    def __init__(self, mu=4., sigma=0.5):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range=8.):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
