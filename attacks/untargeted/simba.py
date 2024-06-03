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
import os

class SimBAttack:
    def __init__(self, 
                eps: float,
                hamming_threshold: int,
                l2_threshold: int,
                max_steps: int,
                fast: bool):
        self.eps = eps
        self.hamming_threshold = hamming_threshold
        self.l2_threshold = l2_threshold
        self.max_steps = max_steps
        self.fast = fast
    
    def sample_pixel(self, img: np.array) -> Tuple[int, int, int]:
        H, W, C = img.shape
        X = int(W*np.random.random(1))
        Y = int(H*np.random.random(1))
        Z = int(C*np.random.random(1))
        return (X, Y, Z)

    def simba(self,
            img: np.array,
            stepsize: int,
            init_hash: int,
            fast: bool,
            hash_func: str) -> Tuple[np.array, int]:
        # Create the additive and subtractive images
        add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
        X, Y, Z = self.sample_pixel(img)
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
        img_hash = utils.compute_hash(img, hash_func=hash_func)
        add_hash = utils.compute_hash(add_img, hash_func=hash_func)
        sub_hash = utils.compute_hash(sub_img, hash_func=hash_func)
        # Compute hamming distances
        img_hamm = utils.distance(init_hash, img_hash, 'hamming', hash_func=hash_func)
        add_hamm = utils.distance(init_hash, add_hash, 'hamming', hash_func=hash_func)
        sub_hamm = utils.distance(init_hash, sub_hash, 'hamming', hash_func=hash_func)

        # Find the image with the largest hamming distance and return it
        hamming_distances = [img_hamm, add_hamm, sub_hamm]
        images = [img, add_img, sub_img]
        idx = np.argmax(hamming_distances) if max(hamming_distances) != min(hamming_distances) else random.randint(0, 2)
        return (images[idx], hamming_distances[idx], 3)

    def attack(self, img_path: str, hash_func: str) -> Tuple[str, int]:
        # Initialize the image
        #filename, filetype = img_path.split('.')
        img = utils.load_img(img_path).astype(np.float32)
        orig_img = copy.deepcopy(img)
        # Compute the hash of the image
        init_hash = utils.compute_hash(img_path, hash_func=hash_func)
        stepsize = int(255*self.eps)
        step_counter = 0
        # Filename of the final SimBA image
        # Define the filepath
        path = img_path.split('/') 
        path[-1] = f'{img_path.split("/")[3].split(".")[0]}_simba.bmp'
        simba_filename = os.path.sep.join(path)
        print('[INFO] SimBA starting...')
        for _ in range(self.max_steps):
            step_counter += 1
            img, hamming_dist, queries = self.simba(img, stepsize, init_hash, self.fast, hash_func)
            l2_dist = utils.distance(img, orig_img, 'l2')
            print(f'Step: {step_counter} L2 Dist: {l2_dist} Hamming Dist: {hamming_dist}')
            if hamming_dist >= self.hamming_threshold and l2_dist >= self.l2_threshold:
                break
        print('[INFO] SimBA completed...')
        utils.save_img(simba_filename, img)
        simba_queries = 1 + queries*step_counter
        return (simba_filename, simba_queries)