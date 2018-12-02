import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle


def plot_trajectories(trajectories, projection=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, traj in enumerate(trajectories):
        rc = np.random.random(3)
        pos = np.array(traj['position']).T
        x, y, z = pos[:, :-1]
        u, v, w = pos[:, 1:] - pos[:, :-1]
        ax.plot(*pos, color=rc, alpha=0.5)
        ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.3, length=2, normalize=True)
        ax.scatter(*pos, marker='o', color=rc)
        if projection:
            ax.plot(*pos[:2], zs=0, zdir='z', color=rc, alpha=0.5)
            ax.plot(xs=np.zeros(pos[0].shape), ys=pos[1], zs=pos[2], zdir='z', color=rc, alpha=0.5)
            ax.plot(xs=pos[0], ys=np.zeros(pos[1].shape), zs=pos[2], zdir='z', color=rc, alpha=0.5)
    plt.show()


def plot_trajectory(trajectory, projection=True):
    pos = np.array(trajectory['position']).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = pos[:, :-1]
    u, v, w = pos[:, 1:] - pos[:, :-1]
    ax.plot(*pos, alpha=0.5)
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.3, length=2, normalize=True)
    ax.scatter(*pos, marker='o')
    if projection:
        ax.plot(*pos[:2], zs=0, zdir='z', alpha=0.5)
        ax.plot(xs=np.zeros(pos[0].shape), ys=pos[1], zs=pos[2], zdir='z', alpha=0.5)
        ax.plot(xs=pos[0], ys=np.zeros(pos[1].shape), zs=pos[2], zdir='z', alpha=0.5)
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


def label_to_2d_image(labels, alpha=0.5, cmap=None):
    """
    :param labels: (x, y, z), values are label values
    :return: (x, y, rgba)
    """
    labels_2d = labels.max(-1).T
    if cmap:
        rgba = cmap((labels_2d % 10 + 1) * (labels_2d > 0))
    else:
        rgba = cm.tab10((labels_2d % 10 + 1) * (labels_2d > 0))
    rgba[:, :, -1] = alpha
    rgba[np.where(labels_2d == 0)] = np.zeros(4)
    return rgba
