'''
    Usage: python simba.py --batch False --folder_path '../images/' --img_path '../images/1.jpeg' --target_path '../images/2.jpeg'   % Single Image Attack
           python simba.py --batch True --folder_path '../images/'                                   % Batch Image Attack
'''

from datetime import datetime
from data import CIFAR10
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import logging
import utils
import copy

def get_hash_of_imgs(pixels, H, W, C, path, add_img, sub_img, stepsize):
    filename, filetype = path.split('.') 
    # Add value to the pixel and get the hash 
    add_img[H][W][C] = pixels + stepsize
    add_img = np.clip(add_img, 0.0, 255.0)
    utils.save_img(f'{filename}_add.{filetype}', add_img)
    add_hash = utils.compute_hash(f'{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    sub_img[H][W] = pixels - stepsize
    sub_img = np.clip(sub_img, 0.0, 255.0)
    utils.save_img(f'{filename}_sub.{filetype}', sub_img)
    sub_hash = utils.compute_hash(f'{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

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

def simba_attack_image(img_path, target_path, eps, max_steps=5000, mismatched_threshold=1):
    filename, filetype = img_path.split('.')
    target_filename, target_filetype = target_path.split('.')
    # Initialize images
    img = utils.load_img(img_path)
    orig_img, add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)
    init_hash = utils.compute_hash(img_path)
    # Calculate hash of target
    target_hash = utils.compute_hash(target_path)
    variable_hash, closest_hash = init_hash, init_hash
    num_steps, min_difference = 0, np.inf
    stepsize = int(eps*255.0)
    for counter in range(max_steps):
        print(counter)
        num_steps = counter+1
        (pixel, Y, X, Z) = utils.sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_imgs( pixel, 
                                                                                Y, 
                                                                                X, 
                                                                                Z,
                                                                                img_path, 
                                                                                add_img, 
                                                                                sub_img, 
                                                                                stepsize)
        simba_filename = f'{filename}_new.{filetype}'
        if abs(target_hash-additive_hash) < abs(target_hash-subtractive_hash):
            if abs(target_hash-additive_hash) < abs(target_hash-closest_hash):
                dist = np.linalg.norm(add_img/255.0-orig_img/255.0)
                logging.info(f'Steps: {counter+1}\tDist: {dist:.2f}\t Current Hash:{additive_hash}\tTarget Hash: {target_hash}')
                img = add_img
                add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
                variable_hash, closest_hash = additive_hash, additive_hash    
                utils.save_img(simba_filename, add_img)
        elif abs(target_hash-additive_hash) > abs(target_hash-subtractive_hash):
            if abs(target_hash-subtractive_hash) < abs(target_hash-closest_hash):
                dist = np.linalg.norm(sub_img/255.0-orig_img/255.0)
                logging.info(f'Steps: {counter+1}\tDist: {dist:.2f}\t Current Hash:{subtractive_hash}\tTarget Hash: {target_hash}')
                img = sub_img
                add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
                variable_hash, closest_hash = subtractive_hash, subtractive_hash
                utils.save_img(simba_filename, sub_img)
        
        if utils.distance(variable_hash,target_hash, 'hamming') <= mismatched_threshold:
            # Calculate l2 distortion 
            dist = np.linalg.norm(img/255.0-orig_img/255.0)
            # Save the new image
            logger.info(f'Saving {filename}.{filetype} after {counter+1} iterations')
            logger.info(f'L2 Distortion: {dist:.2f} units')
            logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(variable_hash)}\tTarget Hash: {hex(target_hash)}')
            utils.save_img(simba_filename, img)
            break
    print(f'\nThe distortion to the original image is {dist:.2f} units')
    logging.info(f'Execution was aborted after {num_steps}/{max_steps} queries')
    return simba_filename, num_steps

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
    folder_path = utils.move_data_to_temp_ram(folder_path, ram_size_mb=50)

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