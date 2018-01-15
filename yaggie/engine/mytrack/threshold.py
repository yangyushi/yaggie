#!/usr/bin/env python3
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

def test_threshold(image, threshold_methods, *block):
    threshold_result = []
    for method in threshold_methods:
        threshold_result.append([method(image[:, :, :, channel], *block) for channel in range(2)])
    for z in range(160):
        fig, ax = plt.subplots(len(threshold_result), 4)
        for method_index in range(len(threshold_result)):
            ax[method_index, 0].imshow(image[:, :, z, 0])
            ax[method_index, 1].imshow(image[:, :, z, 1])
            for channel in range(2):
                threshold = threshold_result[method_index][channel]
                binary = image[:, :, :, channel] >= threshold_result[method_index][channel]
                binary.astype(int)
                ax[method_index, channel+2].imshow(binary[:, :, z], clim=[0, 1])
                for x in ax.flatten():
                    xlabels = [item.get_text() for item in x.get_xticklabels()]
                    xempty_string_labels = ['']*len(xlabels)
                    x.set_xticklabels(xempty_string_labels)
                    ylabels = [item.get_text() for item in x.get_yticklabels()]
                    yempty_string_labels = ['']*len(ylabels)
                    x.set_yticklabels(yempty_string_labels)
        fig.tight_layout()
        fig.savefig('plane-%s.pdf' % str(z))
        plt.clf()

def hysteresis(image):
    high = filters.threshold_yen(image)
    low = high * 0.9
    hight = (image > high).astype(int)
    lowt = (image > low).astype(int)
    binary = filters.apply_hysteresis_threshold(image, low, high)
    return binary
    
def test_hysteresis(image):
    for z in range(image.shape[-2]):
        fig, ax = plt.subplots(2, 2)
        ax[0][0].imshow(image[:, :, z, 0])
        ax[0][1].imshow(image[:, :, z, 1])
        ax[1][0].imshow(hysteresis(image[:, :, z, 0]))
        ax[1][1].imshow(hysteresis(image[:, :, z, 1]))
        fig.tight_layout()
        fig.savefig('plane-%s.pdf' % str(z))
        plt.clf()
        
if __name__ == "__main__":
    from read import read_pkl
    global_threshold_methods = [filters.threshold_otsu,
                                filters.threshold_yen,
                                filters.threshold_isodata,
                                filters.threshold_li,
                                filters.threshold_mean,
                                filters.threshold_minimum,
                                filters.threshold_triangle
                                ]
                        
    local_threshold_methods = [ filters.threshold_local,
                                filters.threshold_niblack,
                                filters.threshold_sauvola]
    #test_threshold(array, local_threshold_methods, 15)

    array = read_4d('pkl\\30_10_mp-0.pkl')
    test_hysteresis(array)
    plt.show()
