from labelme.utils import shape_to_mask
import json

import os.path as osp
import cv2
import glob
import numpy as np

from tqdm import tqdm

normalize = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(int)
def labelme2images(input_dir, output_dir, extention):
    
    print("Generating dataset")
    
    filenames = glob.glob(osp.join(input_dir, "*.json"))
        
    for filename in tqdm(filenames):
        with open(filename, "r", encoding="utf-8") as f:
            dj = json.load(f)

            shapes = dj['shapes']
            h, w = dj['imageHeight'],dj['imageWidth']
            mask = np.zeros((h, w), dtype=np.uint8)
            for shape in shapes:
                mask += shape_to_mask((h, w), shape['points'], shape_type=None,line_width=1, point_size=1)
            output_filename = osp.join(output_dir, osp.basename(filename).replace(".json", extention))
            cv2.imwrite(output_filename, normalize((mask > 0).astype(int)))


if __name__ =="__main__":
    labelme2images("data/masks_json", "data/masks", extention=".tif")