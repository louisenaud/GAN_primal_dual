"""
Project:    GAN_primal_dual
File:       model.py
Created by: louise
On:         08/03/18
At:         5:01 PM
"""
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size=1000, hidden_size=128, output_size=1):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = self.map1(x)
        out = self.map2(out)
        out = self.map3(out)
        out = self.relu1(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_size=1000, hidden_size=128, output_size=1):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.sig1 = nn.Sigmoid()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = self.map1(x)
        out = self.map2(out)
        out = self.map3(out)
        out = self.sig1(out)
        return out


class GAN(nn.Module):
    def __init__(self, input_size=1000, hidden_size=128, output_size=1):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        super(GAN, self).__init__()
        self.generator = Generator(input_size, hidden_size, output_size)
        self.discriminator = Discriminator(input_size, hidden_size, output_size)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        out = self.map1(x)
        out = self.map2(out)
        out = self.map3(out)
        out = self.sig1(out)
        return out
