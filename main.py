import os, json
from tqdm import tqdm

import cleaner, mask_worker


ROOT_FOLDER = json.load(open('configs.json'))["target_feature"] + "_" + \
              json.load(open('configs.json'))["sample_number"] + "_data"
INPUT_TOMO_IMAGES_FOLDER = json.load(open('configs.json'))["input_tomo_images"]
MASK_IMAGES_FOLDER = os.path.join(ROOT_FOLDER,
                                  json.load(open('configs.json'))["mask_images"])

MODEL_NAME = "MODEL_" + json.load(open('configs.json'))["target_feature"] + ".pth"
PYTHONPATH = "C:/Users/79690/anaconda3/python.exe"


def train():
    scale=0.75 if json.load(open('configs.json'))["device"] == "server" else 0.5
    os.system(f"{PYTHONPATH} c:/Users/79690/Desktop/repos/Pytorch-UNet/train.py -s {scale}")
    os.replace("checkpoints\CP_epoch5.pth", MODEL_NAME)


def predict():
    if not os.path.isdir(MASK_IMAGES_FOLDER):
        os.makedirs(MASK_IMAGES_FOLDER)

    filenames = os.listdir(INPUT_TOMO_IMAGES_FOLDER)
    for fn in tqdm(filenames, desc="predictions"):
        input_name = os.path.join(INPUT_TOMO_IMAGES_FOLDER, fn)
        output_name = os.path.join(MASK_IMAGES_FOLDER, fn)
        os.system(f"{PYTHONPATH} c:/Users/79690/Desktop/repos/Pytorch-UNet/predict.py -i\
                  {input_name} -o {output_name} -m {MODEL_NAME}")


if __name__=="__main__":
    need_train = bool(json.load(open('configs.json'))["train"])
    if need_train:
        train()

    predict()
    cleaner.process()
    mask_worker.apply_mask()