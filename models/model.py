"""
Project:    GAN_primal_dual
File:       model.py
Created by: louise
On:         08/03/18
At:         5:01 PM
"""
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Class for 1D GAN generator.
        :param input_dim:
        :param hidden_dim: int, size of hidden layer
        :param output_dim: int, size of desired output, should be the same as data batches
        """
        super(Generator, self).__init__()

        # Fully-connected layer
        fc = nn.Linear(input_dim, hidden_dim, bias=True)
        # initializer
        nn.init.normal(fc.weight)
        nn.init.constant(fc.bias, 0.0)

        # Hidden layer
        self.hidden_layer = nn.Sequential(
            fc,
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=True)
        # initializer
        nn.init.normal(self.output_layer.weight)
        nn.init.constant(self.output_layer.bias, 0.0)

    def forward(self, x):
        """
        Generate noise data.
        :param x: pytorch Variable
        :return: pytorch Variable
        """
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


class Discriminator(nn.Module):
    """

    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """

        :param input_dim:
        :param hidden_dim: int, size of hidden layer
        :param output_dim: int, size of output
        """
        super(Discriminator, self).__init__()

        # Fully-connected layer
        fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        nn.init.normal(fc1.weight)
        nn.init.constant(fc1.bias, 0.0)

        # Hidden layer
        self.hidden_layer = nn.Sequential(
            fc1,
            nn.ReLU()
        )

        # Fully-connected layer
        fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        nn.init.normal(fc2.weight)
        nn.init.constant(fc2.bias, 0.0)

        # Output layer
        self.output_layer = nn.Sequential(
            fc2,
            nn.Sigmoid()  # Return a probability
        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        h = self.hidden_layer(x)
        out = self.output_layer(h)
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
