# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm_notebook

def remove_background(image_path):
    # in progress...
    print('Figuring it out')


def crop_image(image_path):
    # This function is also not perfect yet.
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    t, bw_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    closing = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, np.ones((3, 3)))
    closing = cv2.bitwise_not(closing)
    
    labels, num = measure.label(closing, background=0, connectivity=2, return_num=True)
    max_size, max_blob = 0, None
    for l in range(1, num+1):
        blob = np.zeros_like(labels)
        blob[labels == l] = 1
        nz = np.count_nonzero(blob)
        if nz > max_size:
            max_size = nz
            max_blob = blob
    assert(max_blob is not None)
    
    x, y = np.nonzero(max_blob)
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    max_blob = max_blob[xmin: xmax, ymin:ymax]
    
    # Resized color image
    img = img[xmin: xmax, ymin:ymax]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img, 'RGB')
    return img


if __name__ == "__main__":
    input_folder = r'./data/mytheresa/mytheresa_raw/'
    output_folder = r'./data/mytheresa/mytheresa_preprocessed/'
    
    for f in tqdm_notebook(os.listdir(input_folder)):
        new_img = crop_image(input_folder + f)
        new_path = os.path.join(output_folder + f) 
        new_img.save(new_path, 'JPEG')