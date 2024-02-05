"""
    Usage: python simba.py --folder_path '../../images/' --img_path '../../images/1.bmp'
"""
import sys
sys.path.append('..')

from data import CIFAR10, IMAGENETTE
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import numpy as np

from typing import Tuple
import argparse
import logging
import random
import utils
import copy

def simba(img: np.array,
          stepsize: int,
          target_hash: int,
          fast: bool) -> Tuple[np.array, int]:
    def sample_pixel(img: np.array) -> Tuple[int, int, int]:
        H, W, C = img.shape
        X = int(W*np.random.random(1))
        Y = int(H*np.random.random(1))
        Z = int(C*np.random.random(1))
        return (X, Y, Z)
    # Create the additive and subtractive images
    add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
    X, Y, Z = sample_pixel(img)
    # Perturb the images
    if not fast:
        pixel = img[X][Y][Z]
        add_img[X][Y][Z] = pixel + stepsize
        sub_img[X][Y][Z] = pixel - stepsize
    else:
        pixels = img[X][Y]
        add_img[X][Y] = pixels + stepsize
        sub_img[X][Y] = pixels - stepsize
    # Clip the values into the range [0, 255]
    add_img = np.clip(add_img, 0.0, 255.0)
    sub_img = np.clip(sub_img, 0.0, 255.0)
    # Compute the hash values of the all images
    img_hash = utils.compute_hash(img)
    add_hash = utils.compute_hash(add_img)
    sub_hash = utils.compute_hash(sub_img)
    # Compute hamming distances to the target images
    img_hamm = utils.distance(target_hash, img_hash, 'hamming')
    add_hamm = utils.distance(target_hash, add_hash, 'hamming')
    sub_hamm = utils.distance(target_hash, sub_hash, 'hamming')

    # Find the image with the largest hamming distance and return it
    hamming_distances = [img_hamm, add_hamm, sub_hamm]
    images = [img, add_img, sub_img]
    idx = np.argmin(hamming_distances) if max(hamming_distances) != min(hamming_distances) else random.randint(0, 2)
    return (images[idx], hamming_distances[idx], 3)

def simba_attack_image(img_path: str,
                       target_img_path: str,
                       eps: float,
                       hamming_threshold: int,
                       l2_threshold: int,
                       max_steps: int,
                       fast: bool) -> Tuple[str, int]:
    # Initialize the image
    filename, filetype = img_path.split('.')
    img = utils.load_img(img_path).astype(np.float32)
    orig_img = copy.deepcopy(img)
    # Compute the hash of the target image
    target_hash = utils.compute_hash(target_img_path)
    stepsize = int(255*eps)
    step_counter = 0
    # Filename of the final SimBA image
    simba_filename = f'{filename}_simba.bmp'
    print('[INFO] SimBA starting...')
    for _ in range(max_steps):
        step_counter += 1
        img, hamming_dist, queries = simba(img, stepsize, target_hash, fast)
        l2_dist = utils.distance(img, orig_img, 'l2')
        print(f'Step: {step_counter} L2 Dist: {l2_dist} Hamming Dist: {hamming_dist}')
        if hamming_dist <= hamming_threshold and l2_dist >= l2_threshold:
            break
    print('[INFO] SimBA completed...')
    utils.save_img(simba_filename, img)
    simba_queries = 1 + queries*step_counter
    return (simba_filename, simba_queries)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", help="Path to the image for the attack")
    ap.add_argument("-f", "--folder_path", required=True, help="Path to the directory containing images")
    args = vars(ap.parse_args())

    # Load and Prepare the Data
    if args['img_path'] is not None:
        img_path = args['img_path']
    if args['folder_path'] is not None:
        folder_path = args['folder_path']
    
    dataset = 'imagenette'    
    if dataset == 'cifar10':
        images = CIFAR10()
    if dataset == 'imagenette':
        images = IMAGENETTE()
    x = images.load()
    images.save_to_disk(x, folder_path, num_images=100)
    
    folder_path = utils.move_data_to_temp_ram(folder_path, ram_size_mb=50)

    # Hyperparams
    epsilon = 0.9
    max_mismatched_bits = 16
    max_steps = 5000

    # Attack NeuralHash
    _, _, _, _, path, filetype = img_path.split('.')
    img_path = path.split('/')
    img_path = f'{folder_path}{img_path[2]}.{filetype}'
    simba_attack_image(img_path=img_path, 
                        eps=epsilon, 
                        mismatched_threshold=max_mismatched_bits, 
                        max_steps=max_steps)