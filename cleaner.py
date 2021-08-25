import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as label_image
from scipy.ndimage.morphology import binary_fill_holes

import data_manager as dm
import config_helper


INPUT_TOMO_IMAGES_FOLDER = config_helper.get_input_tomo_img_folder()
MASK_IMAGES_FOLDER = config_helper.get_mask_img_folder()


def remove_small_elements(bin_img):
    labeled_mask, _ = label_image(bin_img)
    cluster_labels, cluster_sizes = np.unique(labeled_mask,
                                            return_counts=True)
    cluster_labels, cluster_sizes = cluster_labels[1:], cluster_sizes[1:]
    relevant_class_label = cluster_labels[np.argmax(cluster_sizes)]

    return labeled_mask==relevant_class_label


def fill_holes(bin_img):
    return binary_fill_holes(bin_img)


def process():
    filecount = len(os.listdir(MASK_IMAGES_FOLDER))
    for bin_mask, fn in tqdm(dm.load_all_data(MASK_IMAGES_FOLDER), total=filecount, desc="cleaning artefacts"):
        bin_mask_filtered = fill_holes(remove_small_elements(bin_mask))
        dm.save_tif(bin_mask_filtered, fn, MASK_IMAGES_FOLDER)
