from utils import compute_hash, sample_pixel, save_img, load_img

import numpy as np
import torch
import cv2

if __name__ == "__main__":
    img_path = '../images/c1.bmp'
    img = cv2.imread(img_path).astype(np.float32)
    init_hash = compute_hash(img_path)
    stepsize = 0.1
    for i in range(1000):
        (x_i, H, W) = sample_pixel(img)
        # Slightly preturb a pixel additively and get the hash 
        img[H][W] = x_i + stepsize
        save_img('../images/c1_add.bmp', img)
        addititve_hash = compute_hash('../images/c1_add.bmp')
        # Slightly preturb a pixel subtracively and get the hash 
        img[H][W] = x_i - stepsize
        save_img('../images/c1_sub.bmp', img)
        subtractive_hash = compute_hash('../images/c1_sub.bmp')
        if abs(hash-addititve_hash) > abs(hash-subtractive_hash):
            # replace original image by additive image
            break
        else:
            # replace original image by subtractive image
            break

