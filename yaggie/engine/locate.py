#!/usr/bin/env python3
# these three engine find local maxima (brightest spot) in image
# all three engine require one initial parameter, roughly related to the size of the object
# engine.run(3d_array) returns the positions (xyz) of all maxima
import warnings
import numpy as np
from . import mytrack
from . import colloids
try:
    import trackpy
except ModuleNotFoundError:
    pass

class MyEngine():
    def __init__(self, radius):
        self.radius = radius
        self.parameters = {'close_distance': 4, 'threshold': 0.8}
    
    def run(self, data):
        maxima = mytrack.find_maxima(data, self.radius, self.parameters['threshold'])
        if not maxima.any():
            warnings.warn("No maxima found, returning empty array")
            return maxima
        else:
            maxima = mytrack.remove_close_maxima(maxima, self.parameters['close_distance'])
            return maxima

class TrackpyEngine():
    def __init__(self, radius):
        radius = np.array(radius)
        self.diameters = 2 * radius + 1  # ensure it is an odd number
        self.parameters = {  # the default parameters in trackpy
                'minmass': None, 
                'maxsize': None, 
                'separation': None,
                'noise_size': 1, 
                'smoothing_size': None,
                'threshold': None,
                'invert': False,
                'percentile': 64,
                'topn': None, 
                'preprocess': True, 
                'max_iterations': 10,
                'filter_before': None, 
                'filter_after': None,
                'characterize': True
                }

    def run(self, data):
        data = np.moveaxis(data, -1, 0)
        data = np.moveaxis(data, -1, 1)
        maxima = trackpy.feature.locate(data, self.diameters, **self.parameters)
        maxima = np.array(maxima)
        maxima = maxima.T[:3].T
        return maxima

class ColloidsEngine():
    def __init__(self, k):
        self.k = k
        self.parameters = {'Octave0': False, 'nbOctaves': 4}

    def run(self, data):
        data = np.moveaxis(data, -1, 0)
        data = np.moveaxis(data, -1, 1)
        shape = [int(dim) for dim in data.shape]
        finder = colloids.MultiscaleBlobFinder(data.shape, **self.parameters)
        centers = finder(data)
        maxima = centers.T[:3].T
        return maxima
