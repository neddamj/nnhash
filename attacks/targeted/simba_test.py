'''
    Usage: python simba_test.py --batch False --folder_path '../../images/' --img_path '../../images/1.jpeg' --target_path '../../images/2.jpeg'    % Single Image Attack
'''
import sys
sys.path.append('..')

from data import CIFAR10, IMAGENETTE
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import numpy as np

import argparse
import logging
import utils
import copy

def sample_pixel(img, batch=False):
    if not batch:
        (H, W) = img.shape[0], img.shape[1]
        (Y, X, Z) = (int(H*np.random.random(1)), int(W*np.random.random(1)), int(3*np.random.random(1)))
        pixel = img[Y][X][Z]
    else:   
        (Y, X, Z) = (int(512*np.random.random(1)), int(512*np.random.random(1)), int(3*np.random.random(1)))
        pixel = list(map(lambda x: x[Y][X][Z], img))
    return (pixel, Y, X, Z)

def perturb_img(pixels, H, W, C, path, add_img, sub_img, stepsize):
    filename, filetype = path.split('.') 
    # Add value to the pixel and get the hash 
    add_img[H][W] = pixels + stepsize
    add_img = np.clip(add_img, 0.0, 255.0)
    utils.save_img(f'{filename}_add.{filetype}', add_img)
    add_hash = utils.compute_hash(f'{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    sub_img[H][W] = pixels - stepsize
    sub_img = np.clip(sub_img, 0.0, 255.0)
    utils.save_img(f'{filename}_sub.{filetype}', sub_img)
    sub_hash = utils.compute_hash(f'{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

def simba_measure_dist(img_path, target_path, eps, max_steps=10000,  mismatched_threshold=1):
    filename, filetype = img_path.split('.')
    # Initialize images
    img = utils.load_img(img_path)
    target_img = utils.load_img(target_path)
    add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
    init_hash = utils.compute_hash(img_path)
    target_hash = utils.compute_hash(target_path)
    min_diff, i = 0, 0
    stepsize = int(eps*255.0)
    # Lists to store the hamming distances
    add_dist_from_orig, sub_dist_from_orig = [], []
    add_dist_to_target, sub_dist_to_target = [], []
    while i < max_steps:
        i += 1
        (pixel, Y, X, Z) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = perturb_img(pixel, 
                                                                               Y, 
                                                                               X, 
                                                                               Z,
                                                                               img_path, 
                                                                               add_img, 
                                                                               sub_img, 
                                                                               stepsize)
        add_hamm_dist, sub_hamm_dist = utils.distance(init_hash, additive_hash, "hamming")/96, utils.distance(init_hash, subtractive_hash, "hamming")/96
        add_target_hamm, sub_target_hamm = utils.distance(target_hash, additive_hash, "hamming")/96, utils.distance(target_hash, subtractive_hash, "hamming")/96
        add_dist_from_orig.append(add_hamm_dist)
        sub_dist_from_orig.append(sub_hamm_dist)
        add_dist_to_target.append(add_target_hamm)
        sub_dist_to_target.append(sub_target_hamm)
        print(f'Iteration: {i} \nAdd Hash: {hex(additive_hash)} \tOrig Add Hamm Dist: {add_hamm_dist} Target Add Hamm Dist: {add_target_hamm}' +
            f'\nSub Hash: {hex(subtractive_hash)} \tOrig Sub Hamm Dist: {sub_hamm_dist} Target Sub Hamm Dist: {sub_target_hamm}\n')
        simba_filename = f'{filename}_new.{filetype}'
        if add_hamm_dist > sub_hamm_dist:
            if add_hamm_dist > min_diff:
                # Only update the adv image when its hamming dist is greater than that
                # of the previous adv image
                min_diff = add_hamm_dist
                utils.save_img(simba_filename, add_img)
            if add_hamm_dist >= mismatched_threshold:
                # calculate l2 distortion 
                dist = utils.distance(add_img, img, 'l2')
                utils.save_img(simba_filename, add_img)
                break
        elif add_hamm_dist < sub_hamm_dist:
            if sub_hamm_dist > min_diff:
                # Only update the adv image when its hamming dist is greater than that
                # of the previous adv image
                min_diff = sub_hamm_dist
                utils.save_img(simba_filename, sub_img)
            if sub_hamm_dist >= mismatched_threshold:
                # calculate l2 distortion 
                dist = utils.distance(sub_img, img, 'l2')
                utils.save_img(simba_filename, sub_img)
                break
    return (add_dist_from_orig, sub_dist_from_orig), (add_dist_to_target, sub_dist_to_target)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", required=True, help="Set to true if you want operations on multiple images, False otherwise")
    ap.add_argument("-i", "--img_path", help="Path to the image for the single image attack")
    ap.add_argument("-f", "--folder_path", required=True, help="Path to the directory containing images for the batch image attack")
    ap.add_argument("-t", "--target_path", required=True, help="Path to the target image for the single image attack")
    args = vars(ap.parse_args())

    # Load and Prepare the Data
    batch = True if args['batch'] == 'True' else False
    if args['img_path'] is not None:
        img_path = args['img_path']
    if args['folder_path'] is not None:
        folder_path = args['folder_path']
    target_path = args['target_path']
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
    max_mismatched_bits = 40
    max_steps = 10000

    # Configure logging
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logging.basicConfig(filename=f'../../logs/targeted/Eps-{epsilon}_Bits-{max_mismatched_bits}_{dt_string}.log',
                        format='%(asctime)s %(message)s',
                        level='DEBUG',
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    logger.info(f'Epsilon: {epsilon}\tMismatched Bits Threshold: {max_mismatched_bits}\n')

    # Path to input image
    _, _, _, _, path, filetype = img_path.split('.')
    img_path = path.split('/')
    img_path = f'{folder_path}{img_path[2]}.{filetype}'
    #Path to target image
    _, _, _, _, path, filetype = target_path.split('.')
    target_path = path.split('/')
    target_path = f'{folder_path}{target_path[2]}.{filetype}'
    simba_attack_image(img_path=img_path, 
                        target_path=target_path,
                        logger=logger,
                        eps=epsilon, 
                        mismatched_threshold=max_mismatched_bits, 
                        max_steps=max_steps)