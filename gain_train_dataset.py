import os
import random
import numpy as np
from os.path import dirname, realpath

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


ROOT_REPOSITORY_FOLDER_PATH = dirname(realpath(__file__))

try:
    os.makedirs(os.path.join(ROOT_REPOSITORY_FOLDER_PATH, "data", "imgs"))
except:
    pass

try:
    os.makedirs(os.path.join(ROOT_REPOSITORY_FOLDER_PATH, "data", "imgs_png"))
except:
    pass

try:
    os.makedirs(os.path.join(ROOT_REPOSITORY_FOLDER_PATH, "data", "masks"))
except:
    pass

IMGS_PNG_FOLDER = os.path.join(ROOT_REPOSITORY_FOLDER_PATH, "data", "imgs_png")
IMGS_FOLDER = os.path.join(ROOT_REPOSITORY_FOLDER_PATH, "data", "imgs")


def choose_random_N_tiff_images(folder, N):
    img_names = os.listdir(folder)
    return random.sample(img_names, N)


def normalize_8_bit(img):
    normed_img = (img - img.min()) / (img.max() - img.min()) * 255
    return normed_img.astype(int)



if __name__ == "__main__":
    tiff_reconstructed_img_dir = r"C:\Users\ankha\Desktop\diplom\geckons"
    already_placed = os.listdir(IMGS_FOLDER)
    N = 3

    img_tiff_names = choose_random_N_tiff_images(tiff_reconstructed_img_dir, N)
    
    for img_tiff_name in tqdm(img_tiff_names):
        if img_tiff_name not in already_placed:
            img_tiff_path_orig = os.path.join(tiff_reconstructed_img_dir, img_tiff_name)

            img = normalize_8_bit(cv2.imread(img_tiff_path_orig, -1))
            img_png_path = os.path.join(IMGS_PNG_FOLDER, "".join(img_tiff_name.split(".")[:-1])+"."+"png")
            img_tiff_path = os.path.join(IMGS_FOLDER, img_tiff_name)

            cv2.imwrite(img_png_path, img)
            cv2.imwrite(img_tiff_path, img)
