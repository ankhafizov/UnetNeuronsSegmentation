from tqdm import tqdm
import os.path as osp
import glob
import cv2

normalize = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(int)


def norm_img_in_folder(folder_path, ext = "*.tif"):
    
    print("Normalization start")
    
    filenames = glob.glob(osp.join(folder_path, ext))
        
    for filename in tqdm(filenames):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = normalize(img)
        cv2.imwrite(filename, img)


norm_img_in_folder("/home/ankhafizov/Desktop/UnetNeuronsSegmentation/data/imgs")