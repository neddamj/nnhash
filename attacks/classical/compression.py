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
            # BMP to JPEG
            jpeg_path = bmp_to_jpg(image_path)
            jpeg_image = load_img(jpeg_path)
            bmp2jpeg_l2 = distance(image, jpeg_image)
            bmp_hash, jpeg_hash = compute_hash(image_path, hash_func=hash_func), compute_hash(jpeg_path, hash_func=hash_func)
            bmp2jpeg_hamming_distance = distance(bmp_hash, jpeg_hash, 'hamming', hash_func=hash_func)/(256)
            bmp2jpeg_success = (bmp2jpeg_hamming_distance >= hamming_threshold)
            print(f'BMP2JPEG:\nRelative Hamming Distance: {bmp2jpeg_hamming_distance:.4f}')
            
            metrics = {
                'Image Path': [image_path],
                'BMP Hash': [(bmp_hash)],
                'JPEG Hash': [(jpeg_hash)],
                'BMP2JPEG Relative Hamming Dist': [bmp2jpeg_hamming_distance],
                'BMP2JPEG Success': [bmp2jpeg_success],
                'BMP2JPEG L2 Dist': [bmp2jpeg_l2]
            }
            
            df = pd.DataFrame.from_dict(metrics)
            base_path = os.path.sep.join(['metrics', f'{hamming_threshold}'])
            if not os.path.exists(base_path):
                print(base_path)
                os.makedirs(base_path, exist_ok=True)
            file_path = os.path.sep.join([base_path, 'recompression.csv'])
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', index=False, header=False)
            else: 
                df.to_csv(file_path, index=False, header=True)