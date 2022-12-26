import numpy as np
import subprocess
import torch
import cv2

def compute_hash(image_path, return_type='hex'):
    hashing_file_name = './nhcalc'
    output = subprocess.check_output([hashing_file_name, image_path])
    hash = output.strip().split()
    return int(hash[1], 16)

def sample_pixel(img):
    (Y, X) = img.shape[0], img.shape[1]
    (Y, X) = (int(Y*torch.rand(1)), int(X*torch.rand(1)))
    pixel = img[Y][X]
    # if pixel is 0 or 255 choose a new pixel
    while 255.0 in pixel or 0.0 in pixel:
        (Y, X) = (int(Y*torch.rand(1)), int(X*torch.rand(1)))
        pixel = img[Y][X]
    return (pixel, Y, X)
 
def save_img(save_path, img):
    cv2.imwrite(save_path, img)

def load_img(img_path):
    return cv2.imread(img_path)