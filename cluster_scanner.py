import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as label_image
from scipy.ndimage.morphology import binary_fill_holes


VOXEL_SIZE = 9 * 1e-6


def hide_image_elements(img, bin_mask):
    """
    where mask, change pixels' brightness min(img) value
    """
    bin_mask = bin_mask.astype(bool)
    bin_mask = binary_fill_holes(bin_mask)
    return bin_mask * np.min(img) + img * (~bin_mask.astype(bool))


def find_big_ones_clusters(bin_mask,
                           min_cluster_length=None,
                           min_cluster_order=None):
    labeled_mask, _ = label_image(bin_mask)
    cluster_labels, cluster_sizes = np.unique(labeled_mask,
                                               return_counts=True)

    # sort from max to min
    sorted_indexes = np.flip(cluster_sizes.argsort())
    cluster_sizes = cluster_sizes[sorted_indexes]
    cluster_labels = cluster_labels[sorted_indexes]

    min_cluster_length = cluster_sizes[min_cluster_order] if min_cluster_order else min_cluster_length
    big_clusters_indexes = cluster_sizes > min_cluster_length

    big_ones_cluster_labels = cluster_labels[big_clusters_indexes]

    #  TODO:refactor this part
    contour_mask = np.zeros(bin_mask.shape, dtype=bool)
    for label in big_ones_cluster_labels:
        if label == 0:
            continue
        contour_mask = np.logical_or(contour_mask, _create_mask_layer_for_label(labeled_mask, label))

    return contour_mask


def _create_mask_layer_for_label(labeled_img, label):
    mask = np.zeros(labeled_img.shape)
    mask +=  np.where(label == labeled_img, True, False)
    return mask.astype(bool)
