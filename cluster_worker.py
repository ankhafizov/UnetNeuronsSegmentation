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


def remove_big_clusters():
    input_mask_folder_name = config_helper.get_RandomWalker_mask_img_folder()
    mask_big_folder_name = input_mask_folder_name + "_big"

    mask3d_initial_data = dm.load_all_data(input_mask_folder_name)
    mask3d_big_data  = dm.load_all_data(mask_big_folder_name)

    for m_init_meta, m_big_meta in tqdm(zip(mask3d_initial_data, mask3d_big_data)):
        m_init, _ = m_init_meta
        m_big, fn = m_big_meta
        dm.save_tif(m_init - m_big, fn, input_mask_folder_name + "_small")


def separate_clusters_by_size(threshold_cluster_size):
    remove_small_clusters(threshold_cluster_size)
    remove_big_clusters()


if __name__ == "__main__":
    # remove_small_clusters()
    remove_big_clusters()
