'''
    Usage: python3 simba_attack.py --batch False --folder_path '../images/' --img_path '../images/02.jpeg'   % Single Image Attack
           python3 simba_attack.py --batch True --folder_path '../images/'                                   % Batch Image Attack
'''

from utils import compute_hash, sample_pixel, save_img, load_img, resize_imgs, load_img_paths, move_data_to_temp_ram
from datetime import datetime
from PIL import Image
import numpy as np
import argparse
import logging
import copy
import cv2

def get_hash_of_preturbed_imgs(pixels, H, W, path, add_img, sub_img, stepsize):
    filename, filetype = path.split('.') 
    # Add value to the pixel and get the hash 
    add_img[H][W] = pixels + stepsize
    add_img = np.clip(add_img, 0.0, 255.0)
    save_img(f'{filename}_add.{filetype}', add_img)
    add_hash = compute_hash(f'{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    sub_img[H][W] = pixels - stepsize
    sub_img = np.clip(sub_img, 0.0, 255.0)
    save_img(f'{filename}_sub.{filetype}', sub_img)
    sub_hash = compute_hash(f'{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

def get_hash_of_batch(pixels, H, W, paths, add_imgs, sub_imgs, stepsize, batch):
    add_ims, sub_ims = [], []
    add_paths, sub_paths = [], []
    add_hashs, sub_hashs = [], []
    for i, path in enumerate(paths):
        _, _, filename, filetype = path.split('.')
        # Add value to the pixel and get the hash 
        add_imgs[i][H][W] = pixels[i] + stepsize
        add_imgs[i] = np.clip(add_imgs[i], 0.0, 255.0)
        save_img(f'{filename}_add.{filetype}', add_imgs[i])
        add_paths.append(f'{filename}_add.{filetype}')
        add_ims.append(add_imgs[i])
        # Subtract value from the pixel and get the hash 
        sub_imgs[i][H][W] = pixels[i] - stepsize
        sub_imgs[i] = np.clip(sub_imgs[i], 0.0, 255.0)
        save_img(f'{filename}_sub.{filetype}', sub_imgs[i])
        sub_paths.append(f'{filename}_sub.{filetype}')
        sub_ims.append(sub_imgs[i])
    add_hashs.append(compute_hash(add_paths, batch=batch))
    sub_hashs.append(compute_hash(sub_paths, batch=batch))
    return (add_ims, add_hashs, sub_ims, sub_hashs)

def simba_attack_image(img_path, eps):
    filename, filetype = img_path.split('.')
    # Initialize images
    img = load_img(img_path)
    add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
    init_hash = compute_hash(img_path)
    i = 0
    stepsize = int(eps*255.0)
    while True:
        i += 1
        (pixel, Y, X) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_preturbed_imgs(pixel, Y, X, img_path, add_img, sub_img, stepsize)
        print(f'Iteration: {i} \tAdd Hash: {hex(additive_hash)} \tSub Hash: {hex(subtractive_hash)}')
        if abs(init_hash-additive_hash) > abs(init_hash-subtractive_hash):
            # calculate l2 distortion 
            dist = np.linalg.norm((add_img-img)/255)
            # replace original image by additive image and store the new hash
            logger.info(f'Saving Added Image after {i} iterations')
            logger.info(f'L2 Distortion: {dist} units')
            logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(additive_hash)}')
            save_img(f'{filename}_new.{filetype}', add_img)
            break
        elif abs(init_hash-additive_hash) < abs(init_hash-subtractive_hash):
            # calculate l2 distortion 
            dist = np.linalg.norm(sub_img-img/255)
            # replace original image by subtractive image and store the new hash
            logger.info(f'Saving Subrtacted Image after {i} iterations')
            logger.info(f'L2 Distortion: {dist} units')
            logger.info(f'Initial Hash: {hex(init_hash)}\tNew Hash: {hex(subtractive_hash)}')
            save_img(f'{filename}_new.{filetype}', sub_img)
            break
    print(f'\nThe distortion to the original image is {dist:.2f} units')

def simba_attack_batch(folder_path, eps, batch=True):
    img_paths = load_img_paths(folder_path)
    resize_imgs(img_paths, batch=batch)
    imgs = load_img(img_paths, batch=batch)
    add_imgs, sub_imgs = copy.deepcopy(imgs), copy.deepcopy(imgs)
    init_hashes = compute_hash(img_paths, batch=batch)
    i = 0
    stepsize = int(eps*255.0)
    hashes = []
    while True:
        if len(img_paths) == 0:
            break
        i += 1
        (pixels, Y, X) = sample_pixel(imgs, batch=batch)
        (add_imgs, additive_hashes, sub_imgs, subtractive_hashes) = get_hash_of_batch(pixels, Y, X, img_paths, add_imgs, sub_imgs, stepsize, batch)
        nz_adds = np.nonzero(init_hashes-additive_hashes)[0]       # Get the indeices of the nonzero additive hashes
        nz_subs = np.nonzero(init_hashes-subtractive_hashes)[0]    # Get the indeices of the nonzero subtractive hashes
        print(f'Iteration: {i}')
        for index in nz_adds:
            _, _, filename, filetype = img_paths[index].split('.')
            logger.info(f'Saving the {index+1} Additive Image')
            logger.info(f'Old Hash: {init_hashes[index]} \tNew Hash: {additive_hashes[index]}')
            save_img(f'../{filename}_new.{filetype}', add_imgs[index])
            # Save the new hashes
            hashes.append(additive_hashes[index])
            # Remove the hashed image from the caches
            init_hashes = np.delete(init_hashes, index)
            additive_hashes = np.delete(additive_hashes, index)
            subtractive_hashes = np.delete(subtractive_hashes, index)
            img_paths.pop(index)
            add_imgs.pop(index)
            sub_imgs.pop(index)
        for index in nz_subs:
            _, _, filename, filetype = img_paths[index].split('.')
            logger.info(f'[INFO] Saving the {index} Subtractive Image')
            logger.info(f'Old Hash: {init_hashes[index]} \tNew Hash: {subtractive_hashes[index]}')
            save_img(f'../{filename}_new.{filetype}', sub_imgs[index])
            # Save the new hashes
            hashes.append(subtractive_hashes[i])
            # Remove the hashed image from the caches
            init_hashes = np.delete(init_hashes, index)
            additive_hashes = np.delete(additive_hashes, index)
            subtractive_hashes = np.delete(subtractive_hashes, index)
            img_paths.pop(index)
            add_imgs.pop(index)
            sub_imgs.pop(index)
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", required=True, help="Set to true if you want operations on multiple images, False otherwise")
    ap.add_argument("-i", "--img_path", help="Path to the image for the single image attack")
    ap.add_argument("-f", "--folder_path", required=True, help="Path to the directory containing images for the batch image attack")
    args = vars(ap.parse_args())

    batch = True if args['batch'] == 'True' else False
    if args['img_path'] is not None:
        img_path = args['img_path']
    if args['folder_path'] is not None:
        folder_path = args['folder_path']
    folder_path = move_data_to_temp_ram(folder_path)

    # Configure logging
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logging.basicConfig(filename=f'../logs/{dt_string}.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Hyperparams
    epsilon = 0.9
    if not batch:
        _, _, path, filetype = img_path.split('.')
        img_path = path.split('/')
        img_path = f'{folder_path}{img_path[2]}.{filetype}'
        simba_attack_image(img_path, eps=epsilon)
    else:
        simba_attack_batch(folder_path, eps=epsilon)