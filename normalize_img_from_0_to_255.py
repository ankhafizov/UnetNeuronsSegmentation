from tqdm import tqdm
import os.path as osp
import glob
import cv2
import numpy as np
from PIL import Image


normalize = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(int)
imread = lambda path: np.asarray(Image.open(path))


def norm_img_in_folder(folder_path, ext = "*.tif"):
    
    print("Normalization start")

    filenames = glob.glob(osp.join(folder_path, ext))
        
    for filename in tqdm(filenames):
        img = imread(filename)
        img = normalize(img)
        cv2.imwrite(filename, img)


norm_img_in_folder("/home/ankhafizov/Desktop/UnetNeuronsSegmentation/geckos_520-525")