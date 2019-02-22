import numpy as np
from scipy import ndimage
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import random_walker
from numba import jit
from scipy.spatial import ConvexHull
from itertools import product


def join_pairs(pairs):
    if len(pairs) == 0:
        return []
    max_val = np.max(np.hstack(pairs)) + 1
    canvas = np.zeros((max_val, max_val), dtype=int)
    p = np.array(pairs)
    canvas[tuple(p.T)] = 1
    labels, _ = ndimage.label(canvas)
    joined_pairs = []
    for val in set(labels[labels > 0]):
        joined_pair = np.unique(np.vstack(np.where(labels == val)))
        joined_pairs.append(joined_pair)
    return joined_pairs


def is_touching(label_1, label_2):
    box_1 = ndimage.find_objects(label_1 > 0)[0]
    box_2 = ndimage.find_objects(label_2 > 0)[0]
    is_touch = True
    for dim in range(label_1.ndim):
        a1 = box_1[dim].start
        a2 = box_1[dim].stop
        b1 = box_2[dim].start
        b2 = box_2[dim].stop
        is_touch *= (((b2 > a1) and (b1 < a2)) or ((b1 < a2) and (b2 > a1)))
    return is_touch


class LabelEngine():
    def __init__(self, minimum_voxel_number):
        self.mvb = minimum_voxel_number

    def run(self, image):
        labels, num = ndimage.label(image)
        binc = np.bincount(labels.ravel())
        noise_index = np.where(binc < self.mvb)  # remove small labels
        mask = np.in1d(labels, noise_index).reshape(image.shape)
        labels[mask] = 0
        labels, num = ndimage.label(labels)
        return labels


class RandomWalkEngine():
    def __init__(self, threshold):
        self.threshold = threshold
        self.parameters = {}

    def __maxima_to_seed(self, image, maxima):
        seed = np.zeros(image.shape)
        for i, maximum in enumerate(maxima):
            index = tuple([int(np.floor(x)) for x in maximum])
            seed[index] = i + 1
        return seed

    def __accept_seed(self, image, seed):
        if len(seed.shape) == 2:
            return self.__maxima_to_seed(image, seed)
        elif len(seed.shape) == 3:
            return seed
        else:
            raise ValueError("seed (shape: x * y * z) or maxima (shape: number * 3) needed!")

    def run(self, image, seed):
        seed = self.__accept_seed(image, seed)
        if type(self.threshold) == np.ndarray:
            blank_region = np.nonzero(np.logical_not(self.threshold))
        else:
            blank_region = np.where(image < self.threshold)
        seed[blank_region] = -1
        labels = random_walker(image, seed, **self.parameters)
        labels[np.where(labels == -1)] = 0
        return labels


class KMeansEngine():
    def __init__(self, threshold):
        self.threshold = threshold

    def __seed_to_maxima(self, seed):
        maxima = []
        xyz = lambda label: np.array(np.nonzero(label))
        com = lambda xyz: np.mean(xyz, axis=1)  # center of mass
        for value in range(int(seed.rabel().max())):
            label = np.zeros(seed.shape)
            label[np.where(seed == value + 1)] = 1
            maxima.append(com(xyz(label)))
        return np.array(maxima)

    def __accept_maxima(self, maxima):
        if len(maxima.shape) == 2:
            return maxima
        elif len(maxima.shape) == 3:
            return self.__seed_to_maxima(image, maxima)
        else:
            raise ValueError("seed (shape: x * y * z) or maxima (shape: number * 3) needed!")

    def run(self, image, maxima):
        labels = np.zeros(image.shape)
        scatters = np.array(np.where(image > self.threshold)).T  # [(xyz), (xyz), ..., (xyz)]
        maxima = self.__accept_maxima(maxima)
        estimator = KMeans(n_clusters=len(maxima), init=maxima)
        estimator.fit(scatters)
        centers = estimator.cluster_centers_  # new max positions
        label_values = estimator.labels_  # [(label), (label), ..., (label)]
        label_values += 1
        label_values = np.expand_dims(label_values, axis=-1)
        xyz_values = np.concatenate((scatters, label_values), axis=1)
        for value in range(label_values.max() + 1):
            xyz_value = np.array(list(filter(lambda x: x[-1] == value, xyz_values)))
            xyz = tuple(xyz_value.T[:3])
            labels[xyz] = value
        return labels


class CHEFEngine():
    def __init__(self, blur=2, number_threshold=2, N=0):
        self.blur = blur
        self.number_threshold = number_threshold
        self.N = N
        if N > 0:
            self.unit_vectors = self.get_unit_vectors()

    def get_unit_vectors(self):
        # a semi-sphere
        azimuths = np.arange(0, 4 * self.N, 1, dtype=float) / (2 * self.N) * np.pi
        elevations = np.arange(0, self.N, 1, dtype=float) / (2 * self.N) * np.pi
        azimuths = np.expand_dims(azimuths, 0)
        elevations = np.expand_dims(elevations, 1)
        unit_vectors = np.array([
            np.cos(azimuths) * np.sin(elevations),
            np.sin(azimuths) * np.sin(elevations),
            np.ones(azimuths.shape) * np.cos(elevations)
        ])
        return unit_vectors.reshape(3, np.max(azimuths.shape) * np.max(elevations.shape)).T

    def get_ch(self, label):
        """
        get the P(V, N) the origional paper
        according to 10.1109/ICPR.2002.1048295
        """
        in_label = np.array(np.where(label > 0)).T  # (number, dimension)
        in_cube = []
        for dim in range(label.ndim):
            coord_range = range(in_label[:, dim].min(), in_label[:, dim].max() + 1)
            in_cube.append(list(coord_range))
        in_cube = list(product(*in_cube))
        in_cube = np.array(in_cube)  # (number, dimension)
        in_convex_hull = []

        f_min_values, f_max_values = [], []

        for nv in self.unit_vectors:
            products = [nv.dot(p) for p in in_label]
            f_min = np.min(products)
            f_max = np.max(products)
            f_min_values.append(f_min)
            f_max_values.append(f_max)

        for c in in_cube:
            is_in_ch = True
            for i, nv in enumerate(self.unit_vectors):
                p = nv.dot(c)
                is_in_ch *= (p >= f_min_values[i])
                is_in_ch *= (p <= f_max_values[i])
            if is_in_ch:
                in_convex_hull.append(c)
        in_convex_hull = np.array(in_convex_hull).astype(int)
        return in_convex_hull

    def should_merge(self, label_1, label_2):
        in_ch_1 = self.get_ch(label_1)
        in_ch_2 = self.get_ch(label_2)
        for coord_1 in in_ch_1:
            for coord_2 in in_ch_2:
                if np.isclose(coord_1, coord_2).all():
                    return True
        return False

    def combine_labels(self, labels):
        values = np.unique(labels)
        values = values[values > 0]
        to_merge = []
        for i, v1 in enumerate(values[: -1]):
            for v2 in values[i + 1:]:
                label_1 = labels.copy()
                label_2 = labels.copy()
                label_1[labels != v1] = 0
                label_2[labels != v2] = 0
                if is_touching(label_1, label_2):
                    print(v1, v2, np.sum(label_1), np.sum(label_2), end='')
                    print('... is touching!', end='')
                    if self.should_merge(label_1, label_2):
                        to_merge.append((v1, v2))
                        print('... and should merge!')
                    else:
                        print('')
        merged_pairs = join_pairs(to_merge)
        for merged in merged_pairs:
            for value in merged:
                labels[labels == value] = merged[0]  # assign the same value for labels
        return labels

    @jit
    def run(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.float64)
        if image.dtype != np.float64:
            image = image.astype(np.float64)
        # blur the image
        g_image = ndimage.filters.gaussian_filter(image, self.blur)
        # generate Hassian Matrix (3, 3, x, y, z)
        h = []
        for i in range(3):
            h.append([])
            for j in range(3):
                h[i].append(np.gradient(np.gradient(g_image, axis=i), axis=j))
        h = np.array(h)
        p = np.zeros(h.shape[2:])
        for x in range(h.shape[2]):
            for y in range(h.shape[3]):
                for z in range(h.shape[4]):
                    # p = 1 if h[:, :, x, y, z] is negative definite, else 0
                    d11 = h[0, 0, x, y, z]
                    d22 = np.linalg.det(h[0:2, 0:2, x, y, z])
                    d33 = np.linalg.det(h[0:3, 0:3, x, y, z])
                    if (-1 * d11 > 0) and (d22 > 0) and (-1 * d33 > 0):
                        p[x, y, z] = 1
                    else:
                        p[x, y, z] = 0
        le = LabelEngine(self.number_threshold)
        labels = le.run(np.array(p))
        if self.N > 0:
            labels = self.combine_labels(labels)
        return labels
