import numpy as np
from tqdm import tqdm
from skimage.segmentation import random_walker
from porespy.filters import chunked_func
from skimage.filters import median
from skimage.morphology import ball
import time
import os

import data_manager as dm
import config_helper


NUM_OF_TIF_SLICES = 500
count = 0


def binarize_img(im, thrs1, thrs2, max_iter):
    if not np.sum(im - im.min())>0:
        return np.zeros_like(im)
    start_time = time.time()
    markers = np.zeros_like(im)
    markers[im > thrs1] = 1
    markers[im < thrs2] = 2
    if len(np.unique(markers)) < 2:
        return np.zeros_like(im)
    print("markers: ", np.unique(markers), "sum: ", np.sum(im - im.min()))
    t = random_walker(im, markers, beta=100)
    global count
    print("count", count, " of ", max_iter, f" %: {count/max_iter:.2f}" ,"| time (min): ", (time.time() - start_time) / 60)
    count += 1
    
    print("return sum: ", np.sum((t<2).astype(int)))
    return (t<2).astype(int)


def segment_neurons(image_3d, z_range, thrs1 = 0.000266, thrs2 = -1.54e-05):

    start_time = time.time()
    image_3d = image_3d[z_range[0]:z_range[1]]
    shot_names = [dm.generate_tif_tomo_section_name(n) for n in range(*z_range)]
    print(np.sum(image_3d))

    print(image_3d.shape)

    image_3d = median(image_3d, selem=ball(1))

    divs = np.array(image_3d.shape) // np.array([20, 40, 40])
    image_3d = chunked_func(func=binarize_img,
                            im=image_3d,
                            thrs1=thrs1,
                            thrs2=thrs2,
                            max_iter=divs[0]*divs[1]*divs[2],
                            divs=divs,
                            overlap=(3, 5, 5),
                            cores=2)

    save_folder = config_helper.get_root_img_folder()
    save_folder = os.path.join(save_folder, "neurons_binary_mask")
    for img2d, shot_name in tqdm(zip(image_3d, shot_names)):
        print(np.sum(img2d))
        dm.save_tif(img2d, shot_name, save_folder)

    end_time = time.time()
    print("cumulative time (min): ", (end_time-start_time)/60)
