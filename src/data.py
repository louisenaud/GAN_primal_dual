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


class DataDistribution:
    def __init__(self, mu, sigma):
        """
        Data distribution parameters
        :param mu: float >= 0, mean of distribution
        :param sigma: float >0, standard deviation of distribution
        """
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples):
        """
        Sample from data distribution.
        :param num_samples: int, number of desired data points
        :return: numpy array of samples
        """
        samples = np.random.normal(self.mu, self.sigma, num_samples)
        samples.sort()
        return samples


class NoiseDistribution:
    def __init__(self, data_range):
        """
        Uniform distribution; setting range here.
        :param data_range:
        """
        self.data_range = data_range

    def sample(self, num_samples):
        """
        Sample from noise distribution.
        :param num_samples: int, desired number of samples.
        :return:
        """
        offset = np.random.random(num_samples) * 0.01
        samples = np.linspace(-self.data_range, self.data_range, num_samples) + offset
        return samples


class TestSample:
    def __init__(self, discriminator, generator, data, gen, data_range, batch_size, num_samples, num_bins):
        """

        :param discriminator:
        :param generator:
        :param data:
        :param gen:
        :param data_range:
        :param batch_size:
        :param num_samples:
        :param num_bins:
        """
        self.D = discriminator
        self.G = generator
        self.data = data
        self.gen = gen
        self.B = batch_size
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.xs = np.linspace(-data_range, data_range, num_samples)
        self.bins = np.linspace(-data_range, data_range, num_bins)

    def decision_boundary(self):
        """
        Computes decision boundary between real and fake data.
        :return: numpy array
        """
        db = np.zeros((self.num_samples, 1))
        for i in range(self.num_samples // self.B):
            x_ = self.xs[self.B*i:self.B*(i+1)]
            x_ = Variable(torch.FloatTensor(np.reshape(x_, [self.B, 1])))

            db[self.B*i:self.B*(i+1)] = self.D(x_).data.numpy()

        return db

    def data_distribution(self):
        """
        Computes histogram of data points (distribution)
        :return: numpy array
        """
        d = self.data.sample(self.num_samples)
        p_data, _ = np.histogram(d, self.bins, density=True)
        return p_data

    def gen_distribution(self):
        """
        Computes histogram of generated data.
        :return:
        """
        zs = self.xs
        g = np.zeros((self.num_samples, 1))
        for i in range(self.num_samples // self.B):
            z_ = zs[self.B * i:self.B * (i + 1)]
            z_ = Variable(torch.FloatTensor(np.reshape(z_, [self.B, 1])))

            g[self.B * i:self.B * (i + 1)] = self.G(z_).data.numpy()

        p_gen, _ = np.histogram(g, self.bins, density=True)
        return p_gen


