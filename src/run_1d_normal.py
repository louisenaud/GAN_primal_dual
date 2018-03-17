"""
Project:    GAN_primal_dual
File:       run_1d_normal.py
Created by: louise
On:         16/03/18
At:         2:12 PM
"""
import argparse
# 1D Gaussian distribution approximation using GAN
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os

from display import DisplayResults
from data import DataDistribution, NoiseDistribution, TestSample
from models.model import Generator, Discriminator


def parse_args():
    desc = "Pytorch implementation of Primal Dual GAN training"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mu', type=float, default=0.8, help='mean of gaussian distribution')
    parser.add_argument('--std', type=float, default=2., help='std of gaussian distribution')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--max_epoch', type=int, default=1000, help='max # of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='directory to save the trained model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--gen_learning_rate', type=float, default=0.0001)
    parser.add_argument('--dis_learning_rate', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--display_step', type=int, default=1000, help='step for displaying loss')

    return parser


if __name__=='__main__':
    parser = parse_args()
    args = parser.parse_args()
    # Parameters
    mu = args.mu
    sigma = args.std
    data_range = 10
    batch_size = args.batch_size
    num_epochs = args.max_epoch
    input_dim = 1
    hidden_dim = 32
    output_dim = 1
    num_epochs = 100000
    num_epochs_pre = 500
    learning_rate = 0.03

    # Samples
    data = DataDistribution(mu, sigma)
    gen = NoiseDistribution(data_range)

    # Models
    G = Generator(input_dim, hidden_dim, output_dim)
    D = Discriminator(input_dim, hidden_dim, output_dim)

    # Loss function
    criterion = torch.nn.BCELoss()

    # optimizer
    optimizer = torch.optim.SGD(D.parameters(), lr=learning_rate)

    D_pre_losses = []
    num_samples_pre = 5000
    num_bins_pre = 100
    for epoch in range(num_epochs_pre):
        # Generate samples
        d = data.sample(num_samples_pre)
        histc, edges = np.histogram(d, num_bins_pre, density=True)

        # Estimate pdf
        max_histc = np.max(histc)
        min_histc = np.min(histc)
        y_ = (histc - min_histc) / (max_histc - min_histc)
        x_ = edges[1:]

        x_ = Variable(torch.FloatTensor(np.reshape(x_, [num_bins_pre, input_dim])))
        y_ = Variable(torch.FloatTensor(np.reshape(y_, [num_bins_pre, output_dim])))

        # Train model
        optimizer.zero_grad()
        D_pre_decision = D(x_)
        D_pre_loss = criterion(D_pre_decision, y_)
        D_pre_loss.backward()
        optimizer.step()

        # Save loss values for plot
        D_pre_losses.append(D_pre_loss[0].data.numpy())

        if epoch % 100 == 0:
            print(epoch, D_pre_loss.data.numpy())

    # Plot loss
    fig, ax = plt.subplots()
    losses = np.array(D_pre_losses)
    plt.plot(losses, label='Pre-train loss')
    plt.title("Pre-training Loss")
    plt.legend()
    plt.show()

    # Test sample after pre-training
    num_samples = 10000
    num_bins = 20
    sample = TestSample(D, G, data, gen, data_range, batch_size, num_samples, num_bins)

    db_pre_trained = sample.decision_boundary()
    gen_pretrained = sample.gen_distribution()

    # Training GAN
    # Optimizers
    D_optimizer = torch.optim.SGD(D.parameters(), lr=learning_rate)
    G_optimizer = torch.optim.SGD(G.parameters(), lr=learning_rate)

    D_losses = []
    G_losses = []
    for epoch in range(num_epochs):
        ########################################################################################
        # Generate samples
        x_ = data.sample(batch_size)
        x_ = Variable(torch.FloatTensor(np.reshape(x_, [batch_size, input_dim])))
        y_real_ = Variable(torch.ones([batch_size, output_dim]))
        y_fake_ = Variable(torch.zeros([batch_size, output_dim]))
        ########################################################################################
        # Train Discriminator
        # 1: real data
        D_real_decision = D(x_)
        D_real_loss = criterion(D_real_decision, y_real_)

        # 2: generated fake data
        z_ = gen.sample(batch_size)
        z_ = Variable(torch.FloatTensor(np.reshape(z_, [batch_size, input_dim])))
        z_ = G(z_)
        D_fake_decision = D(z_)
        D_fake_loss = criterion(D_fake_decision, y_fake_)

        # Back propagation
        D_loss = D_real_loss + D_fake_loss
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        ########################################################################################
        # Train generator
        z_ = gen.sample(batch_size)
        z_ = Variable(torch.FloatTensor(np.reshape(z_, [batch_size, input_dim])))
        z_ = G(z_)
        D_fake_decision = D(z_)
        G_loss = criterion(D_fake_decision, y_real_)

        # Back propagation
        D.zero_grad()
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # Save loss values for plot
        D_losses.append(D_loss[0].data.numpy())
        G_losses.append(G_loss[0].data.numpy())

        if epoch % 100 == 0:
            print(epoch, D_loss.data.numpy(), G_loss.data.numpy())

    # Test sample after pre-training
    sample = TestSample(D, G, data, gen, data_range, batch_size, num_samples, num_bins)

    db_trained = sample.decision_boundary()
    p_data = sample.data_distribution()
    p_gen = sample.gen_distribution()

    # Plot losses
    fig, ax = plt.subplots()
    D_losses = np.array(D_losses)
    G_losses = np.array(G_losses)
    plt.plot(D_losses, label='Discriminator')
    plt.plot(G_losses, label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()

    # DisplayResults result
    DisplayResults = DisplayResults(num_samples, num_bins, data_range, mu, sigma)
    DisplayResults.plot_result(db_pre_trained, db_trained, p_data, p_gen, gen_pretrained)
