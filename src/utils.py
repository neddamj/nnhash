import numpy as np
import subprocess
import torch
import cv2

def compute_hash(image_path):
    hashing_file_name = './nhcalc'
    output = subprocess.check_output([hashing_file_name, image_path])
    hash = output.strip().split()
    return hex(int(hash[1], 16))

def sample_pixel(img):
    (H, W) = img.shape[0], img.shape[1]
    (H, W) = (int(H*torch.rand(1)), int(W*torch.rand(1)))
    return (img[H][W], H, W)

def save_img(save_path, img):
    np.save(save_path, img)

def load_img(img_path):
    return np.load(img_path)