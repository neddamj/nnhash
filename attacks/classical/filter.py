import sys
sys.path.append('..')

import os
import cv2
import pandas as pd
import numpy as np
from data import CIFAR10, IMAGENETTE
from utils import compute_hash, distance, load_img, save_img
import matplotlib.pyplot as plt

def vignette(im):
    image = np.copy(im)
    image = cv2.resize(image, (480, 480))
    rows, cols = image.shape[:2]
    # Generate vignette mask with gaussian kernels
    kernel_x = cv2.getGaussianKernel(rows, 200)
    kernel_y = cv2.getGaussianKernel(cols, 200)
    kernel = kernel_x * kernel_y.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.copy(image)
    # Apply the mask to each channel of the image
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask
    return cv2.resize(output, (224, 224))

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
            # Vignette filter
            vignette_img = vignette(image)
            img_hash, vig_hash = compute_hash(image_path, hash_func=hash_func), compute_hash(vignette_img, hash_func=hash_func)
            vignette_hamming_distance = distance(img_hash, vig_hash, 'hamming', hash_func=hash_func)/(256)
            vignette_success = (vignette_hamming_distance >= hamming_threshold)
            vignette_l2 = distance(image, vignette_img)
            save_img(os.path.sep.join(['..', '..', 'images', f'{i+1}_filt.bmp']), vignette_img)  
            print(f'BMP2JPEG:\nRelative Hamming Distance: {vignette_hamming_distance:.4f}')
            
            metrics = {
                'Image Path': [image_path],
                'IMG Hash': [(img_hash)],
                'VIGNETTE Hash': [(vig_hash)],
                'VIGNETTE Relative Hamming Dist': [vignette_hamming_distance],
                'VIGNETTE Success': [vignette_success],
                'VIGNETTE L2': [vignette_l2]
            }
            
            df = pd.DataFrame.from_dict(metrics)
            base_path = os.path.sep.join(['metrics', f'{hamming_threshold}'])
            if not os.path.exists(base_path):
                os.makedirs(base_path, exist_ok=True)
            file_path = os.path.sep.join([base_path, 'filter.csv'])
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', index=False, header=False)
            else: 
                df.to_csv(file_path, index=False, header=True)