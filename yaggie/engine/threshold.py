#!/usr/bin/env python3
import numpy as np
import warnings
from scipy import ndimage
from skimage import filters
from skimage.morphology import disk

class BaseEngine():
    def apply_mask(self, image, mask):
        """
        chop blank region if no mask is provided
        """
        roi = ndimage.find_objects(mask)[0]
        self.roi = roi
        return (image * mask)#[roi]

class GlobalEngine(BaseEngine):
    """
    Fast, the quality is bad
    """
    def __init__(self, method=filters.threshold_otsu):
        self.method = method

    def run(self, image):
        threshold = self.method(image)
        self.mask = image > threshold
        self.image = self.apply_mask(image, self.mask)
        return self.image

class HysteresisEngine(BaseEngine):
    """
    This is slightly better than global Otsu threshold
    It is also much faster than Local Otsu threhsold
    """
    def __init__(self, expansion=0.5, high=None):
        self.high = high
        self.expansion = expansion

    def run(self, image):
        if not self.high:
            self.high = filters.threshold_yen(image)
        else:
            self.high = self.high * image.max()
        low = self.high * (1 - self.expansion)
        self.mask = filters.apply_hysteresis_threshold(image, low, self.high)
        self.image = self.apply_mask(image, self.mask)
        return self.image

class LocalOtsuEngine(BaseEngine):
    """
    Best quality, horribly slow, considering save the image after threshold everytime!
    todo: directly apply this in  3D
    """
    def __init__(self, radius):
        self.radius = radius
        self.parameters = {'open_radius':0}

    def run(self, image):
        """
        apply local otsu threshold to every z-stack and re-combine them
        image.shape = (x, y, z)
        """
        image = image.copy()
        image = np.moveaxis(image, -1, 0)
        result = []
        image *= 255
        image = image.astype(np.uint8)
        for i, stack in enumerate(image):
            local_otsu = filters.rank.otsu(stack, disk(self.radius))
            mask = stack > local_otsu
            stack = stack * mask
            if self.parameters['open_radius']:
                stack = ndimage.grey_opening(stack, self.parameters['open_radius'])
            result.append(stack)
        result = np.stack(result, axis=0)
        result = np.moveaxis(result, 0, -1)
        self.mask = result > 0
        self.image = self.apply_mask(result, self.mask)
        return self.image
