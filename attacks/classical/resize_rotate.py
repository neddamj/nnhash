import sys
sys.path.append('..')

import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from data import CIFAR10, IMAGENETTE
from utils import compute_hash, distance, load_img, save_img

def resize(img, size_ratio=0.5):
    H, W, _ = img.shape
    dim = (int(size_ratio*W), int(size_ratio*H))
    new_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return new_img

def rotate(img, angle=90):
    H, W = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (W, H))
    return rotated_image

def flip(img):
    pass

if __name__ == "__main__":
    # Load data to disk
    folder_path = os.path.sep.join(['..', '..', 'images'])
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    if len(os.listdir(folder_path)) == 0:
        dataset = 'imagenette'    
        if dataset == 'cifar10':
            images = CIFAR10()
        if dataset == 'imagenette':
            images = IMAGENETTE()
        x = images.load()
        images.save_to_disk(x, folder_path, num_images=100)
        del(x)
        del(images)

    hash_func = 'pdq'
    hamming_thresholds = [0.1, 0.2, 0.3, 0.4]
    for hamming_threshold in hamming_thresholds:
        for i in range(100):
            image_path = os.path.sep.join(['..', '..', 'images', f'{i+1}.bmp'])
            image = load_img(image_path)
            print(f'\nImage {i}')
            # Resize image
            reseized_img = resize(image, size_ratio=0.5)
            save_img(os.path.sep.join(['..', '..', 'images', f'{i+1}_res.bmp']), reseized_img)
            img_hash, resized_hash = compute_hash(image_path, hash_func=hash_func), compute_hash(reseized_img, hash_func=hash_func)
            resized_hamming_distance = distance(img_hash, resized_hash, 'hamming', hash_func=hash_func)/(256)
            resized_success = (resized_hamming_distance >= hamming_threshold)
            print(f'Resizing:\nRelative Hamming Distance: {resized_hamming_distance:.4f}')
            # Rotate image
            rotated_img = rotate(image, angle=90)
            save_img(os.path.sep.join(['..', '..', 'images', f'{i+1}_rot.bmp']), rotated_img)
            img_hash, rotated_hash = compute_hash(image_path, hash_func=hash_func), compute_hash(rotated_img, hash_func=hash_func)
            rotated_hamming_distance = distance(img_hash, rotated_hash, 'hamming', hash_func=hash_func)/(256)
            rotated_success = (rotated_hamming_distance >= hamming_threshold)
            rotated_l2 = distance(image, rotated_img)
            print(f'Rotating:\nRelative Hamming Distance: {rotated_hamming_distance:.4f}')
            
            metrics = {
                'Image Path': [image_path],
                'IMG Hash': [(img_hash)],
                'Resized Hash': [(resized_hash)],
                'Rotated Hash': [(rotated_hash)],
                'Resized Relative Hamming Dist': [resized_hamming_distance],
                'Rotated Relative Hamming Dist': [rotated_hamming_distance],
                'Resized Success': [resized_success],
                'Rotated Success': [rotated_success],
                'Rotated L2': [rotated_l2]
            }
            
            df = pd.DataFrame.from_dict(metrics)
            base_path = os.path.sep.join(['metrics', f'{hamming_threshold}', os.path.sep])
            if not os.path.exists(base_path):
                os.makedirs(base_path, exist_ok=True)
            file_path = os.path.sep.join([base_path, 'resize_rotate.csv'])
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', index=False, header=False)
            else: 
                df.to_csv(file_path, index=False, header=True)