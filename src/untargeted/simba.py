'''
    Usage: python simba.py --batch False --folder_path '../../images/' --img_path '../../images/1.jpeg'    % Single Image Attack
           python simba.py --batch True --folder_path '../images/'                                   % Batch Image Attack
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

def sample_pixel(img):
    (H, W) = img.shape[0], img.shape[1]
    (Y, X, Z) = (int(H*np.random.random(1)), int(W*np.random.random(1)), int(3*np.random.random(1)))
    pixel = img[Y][X][Z]
    return (pixel, Y, X, Z)

def perturb_img(pixels, H, W, C, path, add_img, sub_img, stepsize, fast=False):
    filename, filetype = path.split('.') 
    if not fast:
        add_img[H][W][C] = pixels + stepsize
        sub_img[H][W][C] = pixels - stepsize
    else:
        add_img[H][W] = pixels + stepsize
        sub_img[H][W] = pixels - stepsize
    # Add value to the pixel and get the hash 
    add_img = np.clip(add_img, 0.0, 255.0)
    utils.save_img(f'{filename}_add.{filetype}', add_img)
    add_hash = utils.compute_hash(f'{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    sub_img = np.clip(sub_img, 0.0, 255.0)
    utils.save_img(f'{filename}_sub.{filetype}', sub_img)
    sub_hash = utils.compute_hash(f'{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

def simba_attack_image(img_path, eps, logger, max_steps=10000,  mismatched_threshold=1, l2_threshold=10, fast=False):
    filename, filetype = img_path.split('.')
    # Initialize images
    img = utils.load_img(img_path)
    add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
    init_hash = utils.compute_hash(img_path)
    min_diff, i = 0, 0
    stepsize = int(eps*255.0)
    while i < max_steps:
        i += 1
        (pixel, Y, X, Z) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = perturb_img(pixel, Y, X, Z, img_path, add_img, sub_img, stepsize, fast)
        add_hamm_dist, sub_hamm_dist = utils.distance(init_hash, additive_hash, "hamming"), utils.distance(init_hash, subtractive_hash, "hamming")
        add_l2_dist, sub_l2_dist = utils.distance(add_img, img, 'l2'), utils.distance(sub_img, img, 'l2')
        print(f'Iteration: {i} \tAdd Hash: {hex(additive_hash)} \tAdd Hamm Dist: {add_hamm_dist} ' +
            f'\tSub Hash: {hex(subtractive_hash)} \tSub Hamm Dist: {sub_hamm_dist}')
        simba_filename = f'{filename}_new.{filetype}'
        if add_hamm_dist > sub_hamm_dist:
            if add_hamm_dist >= mismatched_threshold and add_l2_dist >= l2_threshold:
                # calculate l2 distortion 
                dist = utils.distance(add_img, img, 'l2')
                # replace original image by additive image and store the new hash
                logger.info(f'Saving {filename}_add.{filetype} after {i} iterations')
                logger.info(f'L2 Distortion: {dist:.2f} units')
                logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(additive_hash)}')
                utils.save_img(simba_filename, add_img)
                break
            if add_hamm_dist > min_diff:
                # Only update the adv image when its hamming dist is greater than that
                # of the previous adv image
                min_diff = add_hamm_dist
                utils.save_img(simba_filename, add_img)
        elif add_hamm_dist < sub_hamm_dist:
            if sub_hamm_dist >= mismatched_threshold and sub_l2_dist >= l2_threshold:
                # calculate l2 distortion 
                dist = utils.distance(sub_img, img, 'l2')
                # replace original image by subtractive image and store the new hash
                logger.info(f'Saving {filename}_new.{filetype} after {i} iterations')
                logger.info(f'L2 Distortion: {dist:.2f} units')
                logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(subtractive_hash)}')
                utils.save_img(simba_filename, sub_img)
                break
            if sub_hamm_dist > min_diff:
                # Only update the adv image when its hamming dist is greater than that
                # of the previous adv image
                min_diff = sub_hamm_dist
                utils.save_img(simba_filename, sub_img)
    num_queries = 2*i + 1
    return (simba_filename, num_queries)

######################
## Batch Processing ##
######################
def get_hash_of_batch(pixels, H, W, C, paths, add_imgs, sub_imgs, stepsize, batch):
    add_ims, sub_ims = [], []
    add_paths, sub_paths = [], []
    for i, path in enumerate(paths):
        filename, filetype = path.split('.')
        # Add value to the pixel and get the hash 
        add_imgs[i][H][W][C] = pixels[i] + stepsize
        add_imgs[i] = np.clip(add_imgs[i], 0.0, 255.0)
        utils.save_img(f'{filename}_add.{filetype}', add_imgs[i])
        add_paths.append(f'{filename}_add.{filetype}')
        add_ims.append(add_imgs[i])
        # Subtract value from the pixel and get the hash 
        sub_imgs[i][H][W][C] = pixels[i] - stepsize
        sub_imgs[i] = np.clip(sub_imgs[i], 0.0, 255.0)
        utils.save_img(f'{filename}_sub.{filetype}', sub_imgs[i])
        sub_paths.append(f'{filename}_sub.{filetype}')
        sub_ims.append(sub_imgs[i])
    add_hashs = utils.compute_hash(add_paths, batch=batch)
    sub_hashs = utils.compute_hash(sub_paths, batch=batch)
    return (add_ims, add_hashs, sub_ims, sub_hashs) 

def simba_attack_batch(folder_path, eps, max_steps=5000, mismatched_threshold=1, batch=True):
    img_paths = utils.load_img_paths(folder_path)
    utils.resize_imgs(img_paths, batch=batch)
    imgs = utils.load_img(img_paths, batch=batch)
    add_imgs, sub_imgs = copy.deepcopy(imgs), copy.deepcopy(imgs)
    init_hashes = utils.compute_hash(img_paths, batch=batch)
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
                if utils.distance(init_hashes[index], additive_hashes[index], 'hamming') >= mismatched_threshold:
                    counter += 1
                    # calculate l2 distortion 
                    dist = utils.distance(add_imgs[index], imgs[index], 'l2')
                    filename, filetype = img_paths[index].split('.')
                    logger.info(f'Saving "{filename}_add.{filetype}" after {i+1} iterations')
                    logger.info(f'L2 Distortion: {dist:.2f} units')
                    logger.info(f'Old Hash: {hex(init_hashes[index])} \tNew Hash: {hex(additive_hashes[index])}\n')
                    utils.save_img(f'{filename}_new.{filetype}', add_imgs[index])
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
                if utils.distance(init_hashes[index], subtractive_hashes[index], 'hamming') >= mismatched_threshold:
                    counter += 1
                    # calculate l2 distortion 
                    dist = utils.distance(sub_imgs[index], imgs[index], 'l2')
                    filename, filetype = img_paths[index].split('.')
                    logger.info(f'Saving "{filename}_sub.{filetype}" after {i+1} iterations')
                    logger.info(f'L2 Distortion: {dist:.2f} units')
                    logger.info(f'Old Hash: {hex(init_hashes[index])} \tNew Hash: {hex(subtractive_hashes[index])}\n')
                    utils.save_img(f'{filename}_new.{filetype}', sub_imgs[index])
                    # Save the new hashes and distortions
                    hashes.append(subtractive_hashes[index])
                    distortion.append(dist)
                    # Track the imgs with changed hashes
                    processed.append(f'{filename}.{filetype}')
        pbar.set_description(f'Average Distortion: {0 if len(distortion)==0 else sum(distortion)/len(distortion):.2f} Processed Images: {counter}')
    logging.info(f'Total Steps: {i+1}\t Attack Success Rate: {100*(counter/total_imgs):.2f}%\t \
        Average Distortion: {0 if len(distortion)==0 else sum(distortion)/len(distortion):.2f}\t Processed Images: {counter}')
        
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
    l2_threshold = 20
    max_steps = 5000

    # Configure logging
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logging.basicConfig(filename=f'../../logs/untargeted/Eps-{epsilon}_Bits-{max_mismatched_bits}_{dt_string}.log',
                        format='%(asctime)s %(message)s',
                        level='DEBUG',
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    logger.info(f'Epsilon: {epsilon}\tMismatched Bits Threshold: {max_mismatched_bits}\n')

    # Attack NeuralHash
    if not batch:
        _, _, _, _, path, filetype = img_path.split('.')
        img_path = path.split('/')
        img_path = f'{folder_path}{img_path[2]}.{filetype}'
        simba_attack_image(img_path=img_path, 
                           eps=epsilon, 
                           logger=logger,
                           mismatched_threshold=max_mismatched_bits, 
                           l2_threshold=l2_threshold,
                           max_steps=max_steps,
                           fast=False)
    else:
        simba_attack_batch(folder_path=folder_path, 
                           eps=epsilon, 
                           mismatched_threshold=max_mismatched_bits, 
                           max_steps=max_steps)