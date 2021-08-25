import json
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import data_manager as dm
from scipy.signal import fftconvolve
import numpy as np
from skimage import exposure
from scipy.ndimage import zoom

import config_helper


INPUT_TOMO_IMAGES_FOLDER = config_helper.get_input_tomo_img_folder()
MASK_IMAGES_FOLDER = config_helper.get_mask_img_folder()
OUTPUT_MASKED_IMAGES_FOLDER = config_helper.get_OUTPUT_masked_img_folder()
DEVICE = config_helper.get_device()


def vizualize_mask(section_number):
    """
    Показать изображение томо-сечения и наложить на него маску
    """
    img = dm.get_tif_img2d(os.listdir(INPUT_TOMO_IMAGES_FOLDER)[section_number],
                           INPUT_TOMO_IMAGES_FOLDER)
    mask = dm.get_tif_img2d(os.listdir(MASK_IMAGES_FOLDER)[section_number],
                            MASK_IMAGES_FOLDER)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.imshow(exposure.equalize_adapthist(img), cmap="gray")
    ax.contour(mask, colors="red")

    if DEVICE=="server":
        dm.save_fig(fig, "mask")
    elif DEVICE=="laptop":
        plt.show()

    return fig


def vizualize_mask_3d(axis = 1, mask_only=True):
    """
    Показать изображение сечения по любой из осей декартовых координат.
    Можно наложит маску контуром, а можно просто вывести сечения.
    """
    file_names = os.listdir(INPUT_TOMO_IMAGES_FOLDER)
    z_len = len(file_names)
    x_len, y_len = dm.get_tif_img2d(file_names[0], INPUT_TOMO_IMAGES_FOLDER).shape
    indx = (z_len, x_len, y_len)[axis] // 2

    mask_3d= dm.assemble_3d_img(MASK_IMAGES_FOLDER)
    mask_3d = zoom(mask_3d, 0.5)
    mask_3d = zoom(mask_3d, 2)
    mask_3d_section = mask_3d.take(indx, axis)

    if not mask_only:
        img_3d_section = dm.assemble_3d_img(INPUT_TOMO_IMAGES_FOLDER).take(indx, axis)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        axes[0].imshow(img_3d_section, cmap="gray")
        axes[0].contour(mask_3d_section, colors="red")
        axes[1].imshow(mask_3d_section, cmap="gray")
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask_3d_section, cmap="gray")

    if DEVICE=="server":
        dm.save_fig(fig, "mask")
    elif DEVICE=="laptop":
        plt.show()

    return fig


def _smooth_mask():
    mask_3d_section = dm.assemble_3d_img(MASK_IMAGES_FOLDER).astype(np.uint8)
    selem = np.full((3, 3, 3), 1)
    filt_mask = fftconvolve(mask_3d_section, selem)
    return filt_mask > 0.5


def apply_mask(smooth=False):
    """
    Отмаскировать все томо изображения по маскам, полученным нейронкой.
    """
    filecount = len(os.listdir(MASK_IMAGES_FOLDER))
    if smooth:
        mask_sections = dm.assemble_3d_img(MASK_IMAGES_FOLDER)
    else:
        mask_sections = dm.load_all_data(MASK_IMAGES_FOLDER)
    img_sections = dm.load_all_data(INPUT_TOMO_IMAGES_FOLDER)

    for img_data, mask_data in tqdm(zip(img_sections, mask_sections), 
                                    total=filecount, desc="applying masks"):
        img, fn = img_data 
        mask = mask_data if smooth else mask_data[0]
        dm.save_tif(img*mask, fn, OUTPUT_MASKED_IMAGES_FOLDER)


if __name__ == "__main__":
    # vizualize_mask(992)
    vizualize_mask_3d(mask_only=False)    # m = _smooth_mask()
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(m[500], cmap="gray")
    # plt.show()
