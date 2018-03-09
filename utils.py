"""
Project:    GAN_primal_dual
File:       utils.py
Created by: louise
On:         08/03/18
At:         5:03 PM
"""
import numpy as np
import torch
from torch.autograd import Variable


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)
