'''
    Usage: python3 simba_attack.py --batch False --folder_path '../images/' --img_path '../images/02.jpeg'   % Single Image Attack
           python3 simba_attack.py --batch True --folder_path '../images/'                                   % Batch Image Attack
'''

from utils import compute_hash, sample_pixel, save_img, load_img, resize_imgs, load_img_paths, move_data_to_temp_ram, count_mismatched_bits
from datetime import datetime
from data import CIFAR10
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import logging
import copy
import cv2

def get_hash_of_preturbed_imgs(pixels, H, W, C, path, add_img, sub_img, stepsize):
    filename, filetype = path.split('.') 
    print(add_img[H][W][C], pixels)
    # Add value to the pixel and get the hash 
    add_img[H][W][C] = pixels + stepsize
    add_img = np.clip(add_img, 0.0, 255.0)
    save_img(f'{filename}_add.{filetype}', add_img)
    add_hash = compute_hash(f'{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    sub_img[H][W] = pixels - stepsize
    sub_img = np.clip(sub_img, 0.0, 255.0)
    save_img(f'{filename}_sub.{filetype}', sub_img)
    sub_hash = compute_hash(f'{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

def get_hash_of_batch(pixels, H, W, C, paths, add_imgs, sub_imgs, stepsize, batch):
    add_ims, sub_ims = [], []
    add_paths, sub_paths = [], []
    for i, path in enumerate(paths):
        filename, filetype = path.split('.')
        # Add value to the pixel and get the hash 
        add_imgs[i][H][W][C] = pixels[i] + stepsize
        add_imgs[i] = np.clip(add_imgs[i], 0.0, 255.0)
        save_img(f'{filename}_add.{filetype}', add_imgs[i])
        add_paths.append(f'{filename}_add.{filetype}')
        add_ims.append(add_imgs[i])
        # Subtract value from the pixel and get the hash 
        sub_imgs[i][H][W][C] = pixels[i] - stepsize
        sub_imgs[i] = np.clip(sub_imgs[i], 0.0, 255.0)
        save_img(f'{filename}_sub.{filetype}', sub_imgs[i])
        sub_paths.append(f'{filename}_sub.{filetype}')
        sub_ims.append(sub_imgs[i])
    add_hashs = compute_hash(add_paths, batch=batch)
    sub_hashs = compute_hash(sub_paths, batch=batch)
    return (add_ims, add_hashs, sub_ims, sub_hashs) 

def simba_attack_image(img_path, eps, max_steps=5000, mismatched_threshold=1):
    filename, filetype = img_path.split('.')
    # Initialize images
    img = load_img(img_path)
    add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
    init_hash = compute_hash(img_path)
    i = 0
    stepsize = int(eps*255.0)
    while i < max_steps:
        i += 1
        (pixel, Y, X, Z) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_preturbed_imgs(pixel, 
                                                                                        Y, 
                                                                                        X, 
                                                                                        Z,
                                                                                        img_path, 
                                                                                        add_img, 
                                                                                        sub_img, 
                                                                                        stepsize)
        print(f'Iteration: {i} \tAdd Hash: {hex(additive_hash)} \tSub Hash: {hex(subtractive_hash)}')
        if abs(init_hash-additive_hash) > abs(init_hash-subtractive_hash):
            if count_mismatched_bits(init_hash, additive_hash) >= mismatched_threshold:
                # calculate l2 distortion 
                dist = np.linalg.norm((add_img-img)/255)
                # replace original image by additive image and store the new hash
                logger.info(f'Saving {filename}_add.{filetype} after {i} iterations')
                logger.info(f'L2 Distortion: {dist:.2f} units')
                logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(additive_hash)}')
                save_img(f'{filename}_new.{filetype}', add_img)
                break
        elif abs(init_hash-additive_hash) < abs(init_hash-subtractive_hash):
            if count_mismatched_bits(init_hash, subtractive_hash) >= mismatched_threshold:
                # calculate l2 distortion 
                dist = np.linalg.norm(sub_img-img/255)
                # replace original image by subtractive image and store the new hash
                logger.info(f'Saving {filename}_new.{filetype} after {i} iterations')
                logger.info(f'L2 Distortion: {dist:.2f} units')
                logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(subtractive_hash)}')
                save_img(f'{filename}_new.{filetype}', sub_img)
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

    # Configure logging
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logging.basicConfig(filename=f'../logs/{dt_string}.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

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
    epsilon = 0.8
    max_mismatched_bits = 1
    max_steps = 5000
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