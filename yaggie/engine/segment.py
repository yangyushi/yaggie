import numpy as np
from scipy import ndimage
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import random_walker

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
        labels[np.where(labels==-1)] = 0
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
            label[np.where(seed==value + 1)] = 1
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
            xyz_value = np.array(list(filter(lambda x: x[-1]==value, xyz_values)))
            xyz = tuple(xyz_value.T[:3])
            labels[xyz] = value 
        return labels


class CHEFEngine():
    def __init__(self, blur=2):
        self.blur = blur

    def run(self, image):
        blurred_img = ndimage.filters.gaussian_filter(image, self.blur)
        dim = len(image.shape)  # dimension
        xlim, ylim, zlim = image.shape
        hassian = np.zeros((dim, dim, xlim, ylim, zlim))  # 2x2 or 3x3
        for i in range(dim):
            for j in range(dim):
                hassian[i, j, :, :, :] = np.gradient(np.gradient(blurred_img, axis=i), axis=j)
        p = np.zeros(image.shape)
        for x in range(xlim):
            for y in range(ylim):
                for z in range(zlim):
                    p[x, y, z] = self.is_neg_def(hassian[:, :, x, y, z])
        return p

    @staticmethod
    def is_neg_def(x):
        return np.all(np.linalg.eigvals(x) < 0)
