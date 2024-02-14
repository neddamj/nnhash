import sys
sys.path.append('..')

import os
from PIL import Image
import pandas as pd
import numpy as np
from data import CIFAR10, IMAGENETTE
from utils import compute_hash, distance, load_img

def bmp_to_jpg(image_path):
    filename, _ = os.path.splitext(image_path)
    image = Image.open(image_path)
    assert image.format == 'BMP', f'Please supply a BMP image instead of a {image.format}'
    save_path = f'{filename}.jpg'
    image.save(save_path, 'JPEG')
    return save_path

def jpg_to_gif(image_path):
    filename, _ = os.path.splitext(image_path)
    image = Image.open(image_path)
    assert image.format == 'JPEG', f'Please supply a JPEG image instead of a {image.format}'
    save_path = f'{filename}.gif'
    image.save(save_path, 'GIF')
    return save_path

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
            # BMP to JPEG
            jpeg_path = bmp_to_jpg(image_path)
            jpeg_image = load_img(jpeg_path)
            bmp2jpeg_l2 = distance(image, jpeg_image)
            bmp_hash, jpeg_hash = compute_hash(image_path), compute_hash(jpeg_path)
            bmp2jpeg_hamming_distance = distance(bmp_hash, jpeg_hash, 'hamming')/(4*(len(hex(bmp_hash))-2))
            bmp2jpeg_success = (bmp2jpeg_hamming_distance >= hamming_threshold)
            print(f'BMP2JPEG:\nRelative Hamming Distance: {bmp2jpeg_hamming_distance:.4f}\nHash 1: {bmp_hash}\nHash 2: {jpeg_hash}')
            # JPEG to GIF
            gif_path = jpg_to_gif(jpeg_path)
            gif_image = load_img(gif_path)
            jpeg_hash, gif_hash = compute_hash(jpeg_path), compute_hash(gif_path)
            jpeg2gif_hamming_distance = distance(gif_hash, jpeg_hash, 'hamming')/(4*(len(hex(jpeg_hash))-2))
            jpeg2gif_success = (jpeg2gif_hamming_distance >= hamming_threshold)
            print(f'JPEG2GIF:\nRelative Hamming Distance: {jpeg2gif_hamming_distance:.4f}\nHash 1: {jpeg_hash}\nHash 2: {gif_hash}')
            # BMP to GIF
            bmp2gif_hamming_distance = distance(bmp_hash, gif_hash, 'hamming')/(4*(len(hex(bmp_hash))-2))
            bmp2gif_success = (bmp2gif_hamming_distance >= hamming_threshold)
            print(f'JPEG2GIF:\nRelative Hamming Distance: {bmp2gif_hamming_distance:.4f}\nHash 1: {bmp_hash}\nHash 2: {gif_hash}')

            metrics = {
                'Image Path': [image_path],
                'BMP Hash': [hex(bmp_hash)],
                'JPEG Hash': [hex(jpeg_hash)],
                'GIF Hash': [hex(gif_hash)],
                'BMP2JPEG Relative Hamming Dist': [bmp2jpeg_hamming_distance],
                'BMP2JPEG Success': [bmp2jpeg_success],
                'BMP2JPEG L2 Dist': [bmp2jpeg_l2],
                'JPEG2GIF Relative Hamming Dist': [jpeg2gif_hamming_distance],
                'JPEG2GIF Success': [jpeg2gif_success],
                'BMP2GIF Relative Hamming Dist': [bmp2gif_hamming_distance],
                'BMP2GIF Success': [bmp2gif_success]
            }
            
            df = pd.DataFrame.from_dict(metrics)
            file_path = f'metrics/{hamming_threshold}/recompression.csv'
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', index=False, header=False)
            else: 
                df.to_csv(file_path, index=False, header=True)