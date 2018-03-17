"""
Project:    GAN_primal_dual
File:       display.py
Created by: louise
On:         09/03/18
At:         1:59 PM
"""
import os
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


def plot_distributions_1d(data_samples, decision_boundary, generated_samples, sample_range=8.):
    decision_boundary_x = np.linspace(-sample_range, sample_range, len(decision_boundary))
    p_x = np.linspace(-sample_range, sample_range, len(data_samples))
    f, ax = plt.subplots(1)
    ax.plot(decision_boundary_x, decision_boundary, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, data_samples, label='real data')
    plt.plot(p_x, generated_samples, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_animation(animation_frames, animation_path, sample_range):
    """
    
    :param animation_frames: 
    :param animation_path: 
    :param sample_range: 
    :return: 
    """
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_decision_boundary, = ax.plot([], [], label='decision boundary')
    line_data_samples, = ax.plot([], [], label='real data')
    line_generated_samples, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    decision_boundary, data_samples, _ = animation_frames[0]
    decision_boundary_x = np.linspace(-sample_range, sample_range, len(decision_boundary))
    p_x = np.linspace(-sample_range, sample_range, len(data_samples))

    def init():
        line_decision_boundary.set_data([], [])
        line_data_samples.set_data([], [])
        line_generated_samples.set_data([], [])
        frame_number.set_text('')
        return line_decision_boundary, line_data_samples, line_generated_samples, frame_number

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(animation_frames))
        )
        decision_boundary, data_samples, generated_samples = animation_frames[i]
        line_decision_boundary.set_data(decision_boundary_x, decision_boundary)
        line_data_samples.set_data(p_x, data_samples)
        line_generated_samples.set_data(p_x, generated_samples)
        return line_decision_boundary, line_data_samples, line_generated_samples, frame_number

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(animation_frames),
        blit=True
    )
    anim.save(animation_path, fps=30, extra_args=['-vcodec', 'libx264'])


class DisplayResults:
    def __init__(self, num_samples, num_bins, data_range, mu, sigma):
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.data_range = data_range
        self.mu = mu
        self.sigma = sigma

    def plot_result(self, db_pre_trained, db_trained, p_data, p_gen, gen_pretrained):
        d_x = np.linspace(-self.data_range, self.data_range, len(db_trained))
        p_x = np.linspace(-self.data_range, self.data_range, len(p_data))

        f, ax = plt.subplots(1)
        ax.plot(p_x, gen_pretrained, label='Pre-Trained generated data')
        ax.plot(d_x, db_pre_trained, '--', label='Decision boundary(pre-trained)')
        ax.plot(d_x, db_trained, label='Decision boundary')
        ax.set_ylim(0, max(1, np.max(p_data) * 1.1))
        ax.set_xlim(max(self.mu - self.sigma * 3, -self.data_range * 1.1),
                    min(self.mu + self.sigma * 3, self.data_range * 1.1))
        plt.plot(p_x, p_data, label='Real data')
        plt.plot(p_x, p_gen, label='Generated data')
        plt.title('1D Gaussian Approximation using vanilla GAN: ' + '(mu: %3g,' % self.mu + ' sigma: %3g)' % self.sigma)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend(loc=1)
        plt.grid(True)

        # Save plot
        save_dir = "results/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(save_dir + '1D_Gaussian' + '_mu_%g' % self.mu + '_sigma_%g' % self.sigma + '.png')

        plt.show()
