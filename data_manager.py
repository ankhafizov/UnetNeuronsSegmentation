import os
from PIL import Image

import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATABASE_FOLDER_NAME = 'database'


def get_tif_img2d(file_name, folder_name):
    path = os.path.join(SCRIPT_PATH, folder_name, file_name)

    img2d= np.array(Image.open(path))

    return img2d


def save_tif(img2d, file_name, folder_name):
    save_path = os.path.join(SCRIPT_PATH, folder_name)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, file_name)

    if os.path.isfile(save_path):
        os.remove(save_path)

    img2d = Image.fromarray(img2d)
    img2d.save(save_path+'.tif')


def load_all_data(folder_name):
    filenames = sorted(os.listdir(folder_name))

    for fn in filenames:
        yield get_tif_img2d(fn, folder_name), fn[0:4]


def assemble_3d_img(folder_name):
    data = load_all_data(folder_name)
    img3d = [img2d for img2d, _ in data]

    return np.array(img3d)


def save_fig(figure, name):
    save_path = os.path.join(SCRIPT_PATH, 'plots')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, name)

    if os.path.isfile(save_path):
        os.remove(save_path)
    figure.savefig(save_path)
