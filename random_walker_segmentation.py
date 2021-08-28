import numpy as np
from tqdm import tqdm
from skimage.segmentation import random_walker
from porespy.filters import chunked_func
from skimage.filters import median
from skimage.morphology import ball
import time

import data_manager as dm
import config_helper


count = 0


def binarize_img(im, thrs1, thrs2, max_iter):
    global count
    count += 1

    if not np.sum(im - im.min())>0:
        return np.zeros_like(im)
    start_time = time.time()
    markers = np.zeros_like(im)
    markers[im > thrs1] = 1
    markers[im < thrs2] = 2
    if len(np.unique(markers)) < 3:
        return np.zeros_like(im)
    t = random_walker(im, markers, beta=100)
    print("count", count, " of ", max_iter, f" %: {count/max_iter:.2f}" ,"| time (min): ", (time.time() - start_time) / 60)

    return (t<2).astype(int)


def segment_neurons(mask_folder_name, z_range,
                    thrs1 = 0.000266, thrs2 = -1.54e-05):

    image_3d, tomo_section_filenames =  dm.assemble_3d_img_stack(mask_folder_name,
                                                                 z_range)

    image_3d = np.array(image_3d)
    image_3d = median(image_3d, selem=ball(1))

    calc_volume_shape = [20, 40, 40]
    divs = np.array(image_3d.shape) // np.array(calc_volume_shape)
    image_3d = chunked_func(func=binarize_img,
                            im=image_3d,
                            thrs1=thrs1,
                            thrs2=thrs2,
                            max_iter=divs[0]*divs[1]*divs[2],
                            divs=divs,
                            overlap=(3, 5, 5),
                            cores=2)

    normalize = lambda x: (x - x.min()) / (x.max() - x.min())
    image_3d = normalize(image_3d).astype(np.int8)

    save_folder = config_helper.get_RandomWalker_mask_img_folder()
    for img_2d_bin, shot_name in tqdm(zip(image_3d, tomo_section_filenames)):
        dm.save_tif(img_2d_bin, shot_name, save_folder)
    
    global count
    count = 0
