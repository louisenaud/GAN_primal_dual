"""
Project:    GAN_primal_dual
File:       main.py
Created by: louise
On:         08/03/18
At:         1:04 PM
"""
import os, argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data import exp1, get_distribution_sampler, get_generator_input_sampler
from models.model import Generator, Discriminator
from utils import stats, extract, decorate_with_diffs


def parse_args():
    desc = "Pytorch implementation of Primal Dual GAN training"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mu', type=float, default=3., help='mean of gaussian distribution')
    parser.add_argument('--std', type=float, default=2.25, help='std of gaussian distribution')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--max_epoch', type=int, default=50000, help='max # of epochs')
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

    return check_args(parser.parse_args())


def check_args(args):
    """

    :param args:
    :return:
    """
    # directory to save the trained model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # directory to save images
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


if __name__ == "__main__":
    args = parse_args()
    exp1()

    # Normal distribution parameters
    mu = args.mu
    std = args.std
    
    # Model params
    g_input_size = 1     # Random noise dimension coming into generator, per output vector
    g_hidden_size = 50   # Generator complexity
    g_output_size = 1    # size of generated output vector
    d_input_size = 100   # Minibatch size - cardinality of distributions
    d_hidden_size = 50   # Discriminator complexity
    d_output_size = 1    # Single dimension for 'real' vs. 'fake'
    batch_size = d_input_size
    
    dis_learning_rate = args.dis_learning_rate
    gen_learning_rate = args.gen_learning_rate
    optim_betas = (0.9, 0.999)
    num_epochs = args.max_epoch
    print_interval = 200
    d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
    g_steps = 1
    
    # ### Uncomment only one of these
    #(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
    (name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
    
    print("Using data [%s]" % (name))

    d_sampler = get_distribution_sampler(mu, std)  # target distribution
    gi_sampler = get_generator_input_sampler()  # uniform data generator

    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

    d_optimizer = optim.Adam(D.parameters(), lr=dis_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=gen_learning_rate, betas=optim_betas)
    
    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()
    
            #  1A: Train Discriminator on real data
            d_real_data = Variable(d_sampler(d_input_size))
            d_real_decision = D(preprocess(d_real_data))
            d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
            d_real_error.backward()
    
            #  1B: Train D on fake
            d_gen_input = Variable(gi_sampler(batch_size, g_input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
    
        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()
    
            gen_input = Variable(gi_sampler(batch_size, g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))
            g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine
    
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
    
        if epoch % args.display_step == 0:
            print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                                extract(d_real_error)[0],
                                                                extract(d_fake_error)[0],
                                                                extract(g_error)[0],
                                                                stats(extract(d_real_data)),
    stats(extract(d_fake_data))))
