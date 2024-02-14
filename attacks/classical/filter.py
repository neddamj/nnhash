import sys
sys.path.append('..')

import os
import cv2
import pandas as pd
import numpy as np
from data import CIFAR10, IMAGENETTE
from utils import compute_hash, distance, load_img

def vignette(image):
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
    return output

def brightness(image, contrast=1.2, brightness=5.0):
    bright_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return bright_image

if __name__ == "__main__":
    # Load data to disk
    folder_path = '../../images/'
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

    hamming_thresholds = [0.1, 0.2, 0.3, 0.4]
    for hamming_threshold in hamming_thresholds:
        for i in range(100):
            image_path = f'../../images/{i+1}.bmp'
            image = load_img(image_path)
            print(f'\nImage {i}')
            # Vignette filter
            vignette_img = vignette(image)
            img_hash, vig_hash = compute_hash(image_path), compute_hash(vignette_img)
            vignette_hamming_distance = distance(img_hash, vig_hash, 'hamming')/(4*(len(hex(img_hash))-2))
            vignette_success = (vignette_hamming_distance >= hamming_threshold)
            vignette_l2 = distance(image, vignette_img)
            print(f'BMP2JPEG:\nRelative Hamming Distance: {vignette_hamming_distance:.4f}\nHash 1: {img_hash}\nHash 2: {vig_hash}')
            # Change brightness
            bright_img = brightness(image)
            bright_hash = compute_hash(bright_img)
            bright_l2 = distance(bright_img, image)
            bright_hamming_distance = distance(img_hash, bright_hash, 'hamming')/(4*(len(hex(img_hash))-2))
            bright_success = (bright_hamming_distance >= hamming_threshold)
            print(f'JPEG2GIF:\nRelative Hamming Distance: {bright_hamming_distance:.4f}\nHash 1: {img_hash}\nHash 2: {bright_hash}')
            
            metrics = {
                'Image Path': [image_path],
                'IMG Hash': [hex(img_hash)],
                'VIGNETTE Hash': [hex(vig_hash)],
                'BRIGHTNESS Hash': [hex(bright_hash)],
                'VIGNETTE Relative Hamming Dist': [vignette_hamming_distance],
                'BRIGHT Relative Hamming Dist': [bright_hamming_distance],
                'VIGNETTE Success': [vignette_success],
                'VIGNETTE L2': [vignette_l2], 
                'BRIGHT Success': [bright_success],
                'BRIGHT L2': [bright_l2]
            }
            
            df = pd.DataFrame.from_dict(metrics)
            file_path = f'metrics/{hamming_threshold}/filter.csv'
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', index=False, header=False)
            else: 
                df.to_csv(file_path, index=False, header=True)    
