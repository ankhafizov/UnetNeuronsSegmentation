from scipy.ndimage import label
import data_manager as dm
import numpy as np
import cluster_scanner as cs
import config_helper
from tqdm import tqdm


Z_RANGES = [[i*100, i*100+100] for i in range(20)] + [[2000, 2120]]


def remove_small_clusters(threshold_cluster_size = 7000):
    input_mask_folder_name = config_helper.get_RandomWalker_mask_img_folder()
    save_folder = input_mask_folder_name + "_big"

    for rng in tqdm(Z_RANGES):
        mask3d, file_names = dm.assemble_3d_img_stack(input_mask_folder_name, rng)

        mask3d_filtered = cs.find_big_ones_clusters(mask3d,
                                                    min_cluster_length=threshold_cluster_size,
                                                    min_cluster_order=None)
        mask3d_filtered = mask3d_filtered.astype(np.uint8)

        for fn, mask2d in zip(file_names, mask3d_filtered):
            dm.save_tif(mask2d, fn, save_folder)


def remove_big_clusters(threshold_cluster_size = 7000):
    input_mask_folder_name = config_helper.get_RandomWalker_mask_img_folder()
    save_folder = input_mask_folder_name + "_small"

    for rng in tqdm(Z_RANGES):
        mask3d_initial, _ = dm.assemble_3d_img_stack(input_mask_folder_name, rng)
        mask3d_big_folder = config_helper.get_RandomWalker_mask_img_folder() + "_big"
        mask3d_big, file_names  = dm.assemble_3d_img_stack(mask3d_big_folder, rng)

        mask3d_big = remove_small_clusters(threshold_cluster_size)
        mask_dif = mask3d_initial - mask3d_big

        for fn, mask2d in zip(file_names, mask_dif):
            dm.save_tif(mask2d, fn, save_folder)


if __name__ == "__main__":
    remove_small_clusters()
    remove_big_clusters()
