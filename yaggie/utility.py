import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def plot_trajectories(trajectories, projection=True):
    color_list = ['r', 'g', 'b', 'c', 'm', 'y']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax._axis3don = False
    for i, traj in enumerate(trajectories):
        x, y, z = traj[1:].T
        u, v, w = traj[:-1].T - traj[1:].T
        ax.plot(*traj.T, color='c', alpha=0.5)
        ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.3, length=2, normalize=True)
        ax.scatter(*traj.T, marker='o', color=color_list[i % 6])
        if projection:
            ax.plot(*traj.T[:2], zs=0, zdir='z', color=color_list[i % 6], alpha=0.5)
            ax.plot(xs=np.zeros(traj.T[0].shape), ys=traj.T[1], zs=traj.T[2], zdir='z', color=color_list[i % 6], alpha=0.5)
            ax.plot(xs=traj.T[0], ys=np.zeros(traj.T[1].shape), zs=traj.T[2], zdir='z', color=color_list[i % 6], alpha=0.5)
    plt.show() 

def plot_trajectory(trajectory, projection=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = trajectory[1:].T
    u, v, w = trajectory[:-1].T - trajectory[1:].T
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.3, length=2, normalize=True)
    ax.scatter(*trajectory.T, marker='o')
    if projection:
        ax.plot(*trajectory.T[:2], zs=0, zdir='z', alpha=0.5)
        ax.plot(xs=np.zeros(trajectory.T[0].shape), ys=trajectory.T[1], zs=trajectory.T[2], zdir='z', alpha=0.5)
        ax.plot(xs=trajectory.T[0], ys=np.zeros(trajectory.T[1].shape), zs=trajectory.T[2], zdir='z', alpha=0.5)
    plt.show()

def load_large_pkl(fn):
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(fn)
    with open(fn, 'rb') as f:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
    return bytes_in 

def read_pkl(fn):
    try:
        with open(fn, 'rb') as f:
            array = pickle.load(f)
        return array
    except OSError:
        print('lading large pkl...')
        bytes_in = load_large_pkl(fn)
        array = pickle.loads(bytes_in)
        return array

def file_iter(folder, keywords):
    for name in os.listdir(folder):
        if '.' in name:  # execlude folder names
            if np.all([c in name for c in keywords]):
                yield name

def refine_maxima(maxima, image, radius=4, iteration=10):
    """
    implementation refine algorithm from Crocker & Grier 1996
    """
    for i in range(iteration):
        result = []
        for maximum in maxima:
            m0 = 0
            m1 = 0
            # this is stupid
            for i in np.arange(-radius, radius, 1):
                for j in np.arange(-radius, radius, 1):
                    for k in np.arange(-radius, radius, 1):
                        if np.linalg.norm([i, j, k]) < radius:
                            position = maximum + np.array([i, j, k])
                            position = position.astype(int)
                            m1 += np.array([i, j, k]) * image[tuple(position)]
                            m0 += image[tuple(position)]
            delta = m1 / m0
            delta[np.where(delta > 1)] = 1
            delta[np.where(delta < -1)] = -1
            result.append(maximum + delta)
        maxima = np.array(result)
    return np.array(result)
