import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import data_manager as dm
from skimage import exposure

import config_helper


INPUT_TOMO_IMAGES_FOLDER = config_helper.get_input_tomo_img_folder()
MASK_IMAGES_FOLDER = config_helper.get_mask_img_folder()
OUTPUT_MASKED_IMAGES_FOLDER = config_helper.get_OUTPUT_masked_img_folder()
DEVICE = config_helper.get_device()


def _get_section_img(folder, section_number):
    list_dir = sorted(os.listdir(folder))
    return dm.get_tif_img2d(list_dir[section_number], folder)


def vizualize_mask_CNN(section_number):
    """
    Показать изображение томо-сечения и наложить на него маску полученную CNN
    после обработки с параметрами из конфигов.
    """
    img = _get_section_img(INPUT_TOMO_IMAGES_FOLDER, section_number)
    mask = _get_section_img(MASK_IMAGES_FOLDER, section_number)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.imshow(exposure.equalize_adapthist(img), cmap="gray")
    ax.contour(mask, colors="red")

    if DEVICE=="server":
        dm.save_fig(fig, "mask")
    elif DEVICE=="laptop":
        plt.show()

    return fig


def vizualize_mask_3d_CNN(axis = 1, mask_only=True):
    """
    Показать изображение сечения по любой из осей декартовых координат.
    Можно наложит маску контуром, а можно просто вывести сечения.
    """
    file_names = os.listdir(INPUT_TOMO_IMAGES_FOLDER)
    z_len = len(file_names)
    x_len, y_len = dm.get_tif_img2d(file_names[0], INPUT_TOMO_IMAGES_FOLDER).shape
    indx = (z_len, x_len, y_len)[axis] // 2

    mask_3d= dm.assemble_3d_img(MASK_IMAGES_FOLDER)
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


def apply_mask_CNN(smooth=False):
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


def vizualize_mask_RandomWalker(section_number, separate_small_and_big=True):
    tomo_img = _get_section_img(OUTPUT_MASKED_IMAGES_FOLDER, section_number)

    rw_mask_folder = config_helper.get_RandomWalker_mask_img_folder()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))

    tomo_img = exposure.equalize_adapthist(tomo_img,
                                           clip_limit=0.02)
    axes[0].imshow(tomo_img, cmap="gray")
    axes[1].imshow(tomo_img, cmap="gray")
    if separate_small_and_big:
        try:
            big_mask = _get_section_img(rw_mask_folder + "_big", section_number)
            axes[1].contour(big_mask, colors="yellow")
        except FileNotFoundError:
            pass
        try:
            small_mask = _get_section_img(rw_mask_folder + "_small", section_number)
            axes[1].contour(small_mask, colors="red")
        except FileNotFoundError:
            pass
    else:
        mask = _get_section_img(rw_mask_folder, section_number)
        axes[1].contour(mask, colors="red")

    if DEVICE=="server":
        dm.save_fig(fig, "mask_RW")
    elif DEVICE=="laptop":
        plt.show()

    return fig


if __name__ == "__main__":
    section_number = 2100
    vizualize_mask_RandomWalker(section_number)
    # vizualize_mask_3d(mask_only=False)    # m = _smooth_mask()
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(m[500], cmap="gray")
    # plt.show()
