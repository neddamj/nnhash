'''
    Usage: python simba.py --batch False --folder_path '../../images/' --img_path '../../images/1.jpeg' --target_path '../../images/2.jpeg'   % Single Image Attack
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

def get_hash_of_img(pixels, H, W, C, path, add_img, sub_img, stepsize):
    filename, filetype = path.split('.') 
    # Add value to the pixel and get the hash 
    add_img[H][W][C] = pixels + stepsize
    add_img = np.clip(add_img, 0.0, 255.0)
    utils.save_img(f'{filename}_add.{filetype}', add_img)
    add_hash = utils.compute_hash(f'{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    sub_img[H][W][C] = pixels - stepsize
    sub_img = np.clip(sub_img, 0.0, 255.0)
    utils.save_img(f'{filename}_sub.{filetype}', sub_img)
    sub_hash = utils.compute_hash(f'{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

def simba_attack_image(img_path, target_path, eps, logger, max_steps=5000, mismatched_threshold=1):
    filename, filetype = img_path.split('.')
    target_filename, target_filetype = target_path.split('.')
    # Initialize images
    orig_img, target_img = utils.load_img(img_path), utils.load_img(target_path)
    add_img, sub_img = copy.deepcopy(orig_img), copy.deepcopy(orig_img)
    init_hash = utils.compute_hash(img_path)
    # Calculate hash of target
    target_hash = utils.compute_hash(target_path)
    variable_hash, closest_hash = copy.deepcopy(init_hash), copy.deepcopy(init_hash)
    num_steps = 0
    stepsize = int(eps*255.0)
    for counter in range(max_steps):
        num_steps = counter+1
        (pixel, Y, X, Z) = sample_pixel(orig_img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_img(pixel, 
                                                                Y, 
                                                                X, 
                                                                Z,
                                                                img_path, 
                                                                add_img, 
                                                                sub_img, 
                                                                stepsize)
        simba_filename = f'{filename}_new.{filetype}'
        add_hamm_dist = utils.distance(target_hash, additive_hash, 'hamming')
        sub_hamm_dist = utils.distance(target_hash, subtractive_hash, 'hamming')
        closest_hamm_dist = utils.distance(target_hash, closest_hash, 'hamming')
        print(f'{num_steps}| Add Hamming Dist: {add_hamm_dist}\t Sub Hamming Dist: {sub_hamm_dist}')
        if add_hamm_dist < sub_hamm_dist:
            if add_hamm_dist <= closest_hamm_dist:
                logging.info(f'Steps: {counter+1}\t Current Hash:{additive_hash}\tTarget Hash: {target_hash}')
                add_img, sub_img = copy.deepcopy(add_img), copy.deepcopy(add_img)
                variable_hash, closest_hash = additive_hash, additive_hash   
                closest_img = add_img
                utils.save_img(simba_filename, closest_img)
        if add_hamm_dist > sub_hamm_dist:
            if sub_hamm_dist <= closest_hamm_dist:
                logging.info(f'Steps: {counter+1}\t Current Hash:{subtractive_hash}\tTarget Hash: {target_hash}')
                add_img, sub_img = copy.deepcopy(sub_img), copy.deepcopy(sub_img)
                variable_hash, closest_hash = subtractive_hash, subtractive_hash
                closest_img = sub_img
                utils.save_img(simba_filename, closest_img)
        
        if utils.distance(variable_hash,target_hash, 'hamming') <= mismatched_threshold:
            # Calculate l2 distortion 
            dist = utils.distance(closest_img, target_img, 'l2')
            # Save the new image
            logger.info(f'Saving {simba_filename} after {counter+1} iterations')
            logger.info(f'L2 Distortion: {dist:.2f} units')
            logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(variable_hash)}\tTarget Hash: {hex(target_hash)}')
            utils.save_img(simba_filename, closest_img)
            break
    logging.info(f'Execution was aborted after {num_steps}/{max_steps} queries')
    return simba_filename, num_steps
        
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
    epsilon = 0.7
    max_mismatched_bits = 48
    max_steps =10

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