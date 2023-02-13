'''
    Usage: python simba_targeted.py --batch False --folder_path '../images/' --img_path '../images/1.jpeg' --target_path '../images/2.jpeg'   % Single Image Attack
           python simba_targeted.py --batch True --folder_path '../images/'                                   % Batch Image Attack
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

def simba_attack_image(img_path, target_path, eps, max_steps=5000, mismatched_threshold=1):
    filename, filetype = img_path.split('.')
    target_filename, target_filetype = target_path.split('.')
    # Initialize images
    img = load_img(img_path)
    orig_img, add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)
    init_hash = compute_hash(img_path)
    # Calculate hash of target
    target_hash = compute_hash(target_path)
    variable_hash, closest_hash = init_hash, init_hash
    num_steps, min_difference = 0, np.inf
    stepsize = int(eps*255.0)
    for counter in range(max_steps):
        print(counter)
        num_steps = counter+1
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
            if abs(target_hash-additive_hash) < abs(target_hash-closest_hash):
                dist = np.linalg.norm(add_img/255.0-orig_img/255.0)
                logging.info(f'Steps: {counter+1}\tDist: {dist:.2f}\t Current Hash:{additive_hash}\tTarget Hash: {target_hash}')
                img = add_img
                add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
                variable_hash, closest_hash = additive_hash, additive_hash    
                save_img(f'{filename}_new.{filetype}', add_img)
        elif abs(target_hash-additive_hash) > abs(target_hash-subtractive_hash):
            if abs(target_hash-subtractive_hash) < abs(target_hash-closest_hash):
                dist = np.linalg.norm(sub_img/255.0-orig_img/255.0)
                logging.info(f'Steps: {counter+1}\tDist: {dist:.2f}\t Current Hash:{subtractive_hash}\tTarget Hash: {target_hash}')
                img = sub_img
                add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
                variable_hash, closest_hash = subtractive_hash, subtractive_hash
                save_img(f'{filename}_new.{filetype}', sub_img)
        
        if abs(variable_hash-target_hash) <= 2**mismatched_threshold:
            # Calculate l2 distortion 
            dist = np.linalg.norm(img/255.0-orig_img/255.0)
            # Save the new image
            logger.info(f'Saving {filename}.{filetype} after {counter+1} iterations')
            logger.info(f'L2 Distortion: {dist:.2f} units')
            logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(variable_hash)}\tTarget Hash: {hex(target_hash)}')
            save_img(f'{filename}_new.{filetype}', img)
            break
    print(f'\nThe distortion to the original image is {dist:.2f} units')
    logging.info(f'Execution was aborted after {num_steps}/{max_steps} queries')

def simba_attack_batch(folder_path, eps, max_steps=5000, mismatched_threshold=1, batch=True):
    pass
        
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
    cifar10 = CIFAR10()
    (x_train, y_train), (x_test, y_test) = cifar10.load()
    cifar10.save_to_disk(x_train, folder_path, num_images=100)
    folder_path = move_data_to_temp_ram(folder_path, ram_size_mb=50)

    # Hyperparams
    epsilon = 0.2
    max_mismatched_bits = 24
    max_steps = 10000

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
        # Path to input image
        _, _, path, filetype = img_path.split('.')
        img_path = path.split('/')
        img_path = f'{folder_path}{img_path[2]}.{filetype}'
        #Path to target image
        _, _, path, filetype = target_path.split('.')
        target_path = path.split('/')
        target_path = f'{folder_path}{target_path[2]}.{filetype}'
        simba_attack_image(img_path=img_path, 
                           target_path=target_path,
                           eps=epsilon, 
                           mismatched_threshold=max_mismatched_bits, 
                           max_steps=max_steps)
    else:
        simba_attack_batch(folder_path=folder_path, 
                           target_path=target_path,
                           eps=epsilon, 
                           mismatched_threshold=max_mismatched_bits, 
                           max_steps=max_steps)