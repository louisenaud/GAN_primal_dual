"""
Project:    GAN_primal_dual
File:       display.py
Created by: louise
On:         09/03/18
At:         1:59 PM
"""
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

def samples(
    model,
    session,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):

    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = model.discriminator.forward(data)

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = model.discriminator.forward(data)
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg