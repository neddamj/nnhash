'''
    Usage: python3 simba_attack_targeted.py --batch False --folder_path '../images/' --img_path '../images/02.jpeg'   % Single Image Attack
           python3 simba_attack_targeted.py --batch True --folder_path '../images/'                                   % Batch Image Attack
'''

from simba import get_hash_of_batch, get_hash_of_imgs
from datetime import datetime
from data import CIFAR10
from tqdm import tqdm
from PIL import Image
from utils import *
import numpy as np
import argparse
import logging
import copy
import cv2

def simba_attack_image(img_path, target_path, eps, max_steps=5000, mismatched_threshold=1):
    filename, filetype = img_path.split('.')
    target_filename, target_filetype = target_path.split('.')
    # Initialize images
    img = load_img(img_path)
    orig_img, add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)
    init_hash = compute_hash(img_path)
    variable_hash = init_hash
    # Calculate hash of target
    target_hash = compute_hash(target_path)
    counter = 0
    stepsize = int(eps*255.0)
    while True:
        counter += 1
        (pixel, Y, X, Z) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_imgs( pixel, 
                                                                                Y, 
                                                                                X, 
                                                                                Z,
                                                                                img_path, 
                                                                                add_img, 
                                                                                sub_img, 
                                                                                stepsize)
        if abs(target_hash-additive_hash) < abs(target_hash-subtractive_hash):
            print(f'Replacing the previous image with the additive image after {counter} steps')
            img = add_img
            add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
            variable_hash = additive_hash
        elif abs(target_hash-additive_hash) > abs(target_hash-subtractive_hash):
            print(f'Replacing the previous image with the subtractive image after {counter} steps')
            img = sub_img
            add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
            variable_hash = subtractive_hash
        
        if count_mismatched_bits(variable_hash-target_hash) <= mismatched_threshold:
            # Calculate l2 distortion 
            dist = np.linalg.norm(img/255.0-orig_img/255.0)
            # Save the new image
            logger.info(f'Saving {filename}.{filetype} after {counter} iterations')
            logger.info(f'L2 Distortion: {dist:.2f} units')
            logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(variable_hash)}\tTarget Hash: {hex(target_hash)}')
            save_img(f'{filename}_new.{filetype}', add_img)
            break
    print(f'\nThe distortion to the original image is {dist:.2f} units')

def simba_attack_batch(folder_path, eps, max_steps=5000, mismatched_threshold=1, batch=True):
    img_paths = load_img_paths(folder_path)
    resize_imgs(img_paths, batch=batch)
    imgs = load_img(img_paths, batch=batch)
    add_imgs, sub_imgs = copy.deepcopy(imgs), copy.deepcopy(imgs)
    init_hashes = compute_hash(img_paths, batch=batch)
    counter, i = 0, 0
    total_imgs = len(img_paths)
    stepsize = int(eps*255.0)
    hashes, distortion, processed = [], [], []
    pbar = tqdm(range(max_steps))
    for i in pbar:
        if counter == 100:
            break
        (pixels, Y, X, Z) = sample_pixel(imgs, batch=batch)
        (add_imgs, additive_hashes, sub_imgs, subtractive_hashes) = get_hash_of_batch(pixels, 
                                                                                    Y, 
                                                                                    X, 
                                                                                    Z,
                                                                                    img_paths, 
                                                                                    add_imgs, 
                                                                                    sub_imgs, 
                                                                                    stepsize, 
                                                                                    batch)
        nz_adds = np.nonzero(init_hashes-additive_hashes)[0]       # Get the indices of the nonzero additive hashes
        nz_subs = np.nonzero(init_hashes-subtractive_hashes)[0]    # Get the indices of the nonzero subtractive hashes
        for index in nz_adds:
            filename, filetype = img_paths[index].split('.')
            if f'{filename}.{filetype}' in processed:
                continue
            else:
                if count_mismatched_bits(init_hashes[index], additive_hashes[index]) >= mismatched_threshold:
                    counter += 1
                    # calculate l2 distortion 
                    dist = np.linalg.norm(add_imgs[index]/255.0-imgs[index]/255.0)
                    filename, filetype = img_paths[index].split('.')
                    logger.info(f'Saving "{filename}_add.{filetype}" after {i+1} iterations')
                    logger.info(f'L2 Distortion: {dist:.2f} units')
                    logger.info(f'Old Hash: {hex(init_hashes[index])} \tNew Hash: {hex(additive_hashes[index])}\n')
                    save_img(f'{filename}_new.{filetype}', add_imgs[index])
                    # Save the new hashes and distortions
                    hashes.append(additive_hashes[index])
                    distortion.append(dist)
                    # Track the imgs with changed hashes
                    processed.append(f'{filename}.{filetype}')
        for index in nz_subs:
            filename, filetype = img_paths[index].split('.')
            if f'{filename}.{filetype}' in processed:
                continue
            else:
                if count_mismatched_bits(init_hashes[index], subtractive_hashes[index]) >= mismatched_threshold:
                    counter += 1
                    # calculate l2 distortion 
                    dist = np.linalg.norm(sub_imgs[index]/255.0-imgs[index]/255.0)
                    filename, filetype = img_paths[index].split('.')
                    logger.info(f'Saving "{filename}_sub.{filetype}" after {i+1} iterations')
                    logger.info(f'L2 Distortion: {dist:.2f} units')
                    logger.info(f'Old Hash: {hex(init_hashes[index])} \tNew Hash: {hex(subtractive_hashes[index])}\n')
                    save_img(f'{filename}_new.{filetype}', sub_imgs[index])
                    # Save the new hashes and distortions
                    hashes.append(subtractive_hashes[index])
                    distortion.append(dist)
                    # Track the imgs with changed hashes
                    processed.append(f'{filename}.{filetype}')
        pbar.set_description(f'Average Distortion: {0 if len(distortion)==0 else sum(distortion)/len(distortion):.2f} Processed Images: {counter}')
    logging.info(f'Total Steps: {i+1}\t Attack Success Rate: {100*(counter/total_imgs):.2f}%\t \
        Average Distortion: {sum(distortion)/len(distortion):.2f}\t Processed Images: {counter}')
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", required=True, help="Set to true if you want operations on multiple images, False otherwise")
    ap.add_argument("-i", "--img_path", help="Path to the image for the single image attack")
    ap.add_argument("-f", "--folder_path", required=True, help="Path to the directory containing images for the batch image attack")
    args = vars(ap.parse_args())

    # Load and Prepare the Data
    batch = True if args['batch'] == 'True' else False
    if args['img_path'] is not None:
        img_path = args['img_path']
    if args['folder_path'] is not None:
        folder_path = args['folder_path']
    cifar10 = CIFAR10()
    (x_train, y_train), (x_test, y_test) = cifar10.load()
    cifar10.save_to_disk(x_train, folder_path, num_images=100)
    folder_path = move_data_to_temp_ram(folder_path, ram_size_mb=50)

    # Hyperparams
    epsilon = 0.4
    max_mismatched_bits = 8
    max_steps = 5000

    # Configure logging
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logging.basicConfig(filename=f'../logs/targeted/Eps-{epsilon}_Bits-{max_mismatched_bits}_{dt_string}.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(f'Epsilon: {epsilon}\tMismatched Bits Threshold: {max_mismatched_bits}\n')

    # Attack NeuralHash
    if not batch:
        _, _, path, filetype = img_path.split('.')
        img_path = path.split('/')
        img_path = f'{folder_path}{img_path[2]}.{filetype}'
        simba_attack_image(img_path=img_path, 
                           eps=epsilon, 
                           mismatched_threshold=max_mismatched_bits, 
                           max_steps=max_steps)
    else:
        simba_attack_batch(folder_path=folder_path, 
                           eps=epsilon, 
                           mismatched_threshold=max_mismatched_bits, 
                           max_steps=max_steps)