#!/usr/bin/env python3
from scipy import ndimage
from scipy.spatial import cKDTree
import numpy as np

def find_maxima(image, radius, intensity=0.5):
    """
    image: input 2d or 3d image, whose shape is (x_size,y_size, z_size)
    radius: the pixel/voxel used when doing grey_dilation, in the form of (x, y, z)
    return: the position of the local maxima, [(x1, y1, z1), (x2, y2, z2) ,... ]
    """
    dilation = ndimage.grey_dilation(image, size=radius, mode="constant")
    maxima = (image==dilation) & (image > intensity)  # same shape as image, (x, y, z)
    max_positions = np.vstack(np.where(maxima))  # shape (3 x feature_number), containing xs, ys and zs
    return max_positions.T

def remove_maxima_near_edge(max_positions, raw_image, edge_margin):
    """
    max_positions: coordinates of all local maximas, [(x1, y1, z1), (x2, y2, z2), ... ]
    return: new local maximas [(x1, y1, z1), (x2, y2, z2), ... ]
    """
    new_positions = max_positions.copy()
    size_xyz = np.array(raw_image.shape)
    near_edge = np.any((new_positions <= edge_margin) | (new_positions >= (size_xyz - edge_margin - 1)), axis=1)
    new_positions = new_positions[~near_edge]
    return new_positions


def refine_maxima(max_positions, radius, raw_image, shift_threshold=0.6, max_movement=10):
    """
    NOT WORKING FOR NOW, TODO: FINISH THIS
    max_positions: coordinates of all local maximas, [(x1, y1, z1), (x2, y2, z2), ... ]
    radius: the (estimated) radius of the feature
    raw_image: nd array
    shift_threshold: if element of off_center is larger than the threshold, then move the maxima
    max_movement: maxima position only allowed to move [max_movement] unit along each axis
    return: new local maximas [(x1, y1, z1), (x2, y2, z2), ... ]
    """
    radius = np.array(radius)
    new_max_positions = max_positions.copy()
    points = [np.arange(-r, r+1) for r in radius]
    temp_coords_for_mask = np.meshgrid(*points, indexing='ij')
    temp_coords_for_mask = [(coord / rad) ** 2 for coord, rad in zip(temp_coords_for_mask, radius)]
    mask = sum(temp_coords_for_mask) <= 1
    grid_for_refine = np.ogrid[[slice(0, i) for i in mask.shape]] 
    for maxima in new_max_positions:
        movement = 0
        adjacent_indice = [slice(x-r, x+r+1) for x, r in zip(maxima, radius)]
        while movement <= max_movement:
            voxels_around = mask * raw_image[adjacent_indice]
            cm_n = [(voxels_around * grid_for_refine[d]).sum() / voxels_around.sum()
                    for d in range(voxels_around.ndim)]  # why this name?
            off_center = cm_n - radius
            if np.all(off_center < shift_threshold):
                movement = max_movement + 1
            else:
                #print("moving maxima from {}".format(maxima), end=' ')
                maxima += np.ones(radius.shape) * off_center > shift_threshold 
                maxima -= np.ones(radius.shape) * off_center < -shift_threshold 
                #print("to {}".format(maxima))
                movement += 1
    return new_max_positions

def remove_close_maxima(max_positions, min_distance): 
    """
    max_positions: coordinates of all local maximas, [(x1, y1, z1), (x2, y2, z2), ... ]
    min_distance: distance between nearest neighbours
    return: new local maximas [(x1, y1, z1), (x2, y2, z2), ... ]
    """
    kdt = cKDTree(max_positions)  # 
    close_pairs = kdt.query_pairs(min_distance)
    if close_pairs == set([]):
        print(type(close_pairs))
        return max_positions
    else:
        index_0 = np.fromiter((x[0] for x in close_pairs), dtype=int)
        index_1 = np.fromiter((x[1] for x in close_pairs), dtype=int)
        to_drop = np.where(np.sum(max_positions[index_0], 1) > 
                           np.sum(max_positions[index_1], 1),
                           index_1, index_0)  # WHY??
        to_drop = np.unique(to_drop)
        new_positions = max_positions.copy()
        new_positions = np.delete(new_positions, to_drop, axis=0)
        return new_positions
