import os

import cleaner, mask_worker
from tqdm import tqdm

import config_helper
import data_manager as dm
import random_walker_segmentation as rws


INPUT_TOMO_IMAGES_FOLDER = config_helper.get_input_tomo_img_folder()
MASK_IMAGES_FOLDER = config_helper.get_mask_img_folder()
MASKED_IMAGES_FOLDER = config_helper.get_OUTPUT_masked_img_folder()

MODEL_NAME = config_helper.get_model_name()

DEVICE = config_helper.get_device()
PYTHONPATH = "python"


def train():
    scale = config_helper.open_config()["scale_img"]
    os.system(f"{PYTHONPATH} train.py -s {scale}")
    os.replace("checkpoints\CP_epoch5.pth", MODEL_NAME)


def predict(beginning=0):
    if not os.path.isdir(MASK_IMAGES_FOLDER):
        os.makedirs(MASK_IMAGES_FOLDER)

    filenames = os.listdir(INPUT_TOMO_IMAGES_FOLDER)
    for i, fn in enumerate(tqdm(filenames, desc="predictions")):
        if i>=beginning:
            input_name = os.path.join(INPUT_TOMO_IMAGES_FOLDER, fn)
            output_name = os.path.join(MASK_IMAGES_FOLDER, fn)
            os.system(f"{PYTHONPATH} predict.py -i\
                    {input_name} -o {output_name} -m {MODEL_NAME}")





if __name__=="__main__":
    if config_helper.does_need_train():
        train()
    if config_helper.does_need_predict():
        predict(config_helper.get_start_prediction_point())
    if config_helper.does_need_cleaning():
        cleaner.process()
    if config_helper.open_config()["apply_masks"]:
        mask_worker.apply_mask()
    if config_helper.open_config()["segment_neurons"]:
        img_metadata = dm.load_all_data(MASKED_IMAGES_FOLDER)
        z_ranges = config_helper.open_config()["z_ranges"]

        for z_range in z_ranges:        
            rws.segment_neurons(img_metadata, z_range,
                                thrs1 = 0.000266, thrs2 = -1.54e-05)