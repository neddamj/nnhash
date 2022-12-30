'''
    Usage: python3 simba_attack.py --img_path '../images/02.jpeg'   % Single Image Attack
           python3 simba_attack.py --folder_path '../images/'       % Batch Image Attack
'''

from utils import compute_hash, sample_pixel, save_img, load_img, resize_imgs, load_img_paths
from PIL import Image
import numpy as np
import argparse
import copy
import cv2

def get_hash_of_preturbed_imgs(pixels, H, W, path, add_img, sub_img, stepsize):
    _, _, filename, filetype = path.split('.') 
    # Add value to the pixel and get the hash 
    for i, pixel in enumerate(pixels):
        if pixel + stepsize <= 255.0:
            add_img[H][W][i] = pixel + stepsize
        else:
            add_img[H][W][i] == 255.0
    save_img(f'../{filename}_add.{filetype}', add_img)
    add_hash = compute_hash(f'../{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    for i, pixel in enumerate(pixels):
        if pixel - stepsize >= 0.0:
            sub_img[H][W][i] = pixel - stepsize
        else:
            sub_img[H][W][i] = 0.0
    save_img(f'../{filename}_sub.{filetype}', sub_img)
    sub_hash = compute_hash(f'../{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

def simba_attack_image(img_path):
    _, _, filename, filetype = img_path.split('.')
    # Initialize images
    img = load_img(img_path)
    add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
    init_hash = compute_hash(img_path)
    epsilon, i = 0.9, 0
    stepsize = int(epsilon*255.0)
    while True:
        i += 1
        (pixel, Y, X) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_preturbed_imgs(pixel, Y, X, img_path, add_img, sub_img, stepsize)
        print(f'Iteration: {i} \tAdd Hash: {hex(additive_hash)} \tSub Hash: {hex(subtractive_hash)}')
        if abs(init_hash-additive_hash) > abs(init_hash-subtractive_hash):
            # calculate l2 distortion 
            dist = np.linalg.norm((add_img-img)/255)
            # replace original image by additive image and store the new hash
            print(f'[INFO] Saving Added Image after {i} iterations')
            save_img(f'../{filename}_new.{filetype}', add_img)
            break
        elif abs(init_hash-additive_hash) < abs(init_hash-subtractive_hash):
            # calculate l2 distortion 
            dist = np.linalg.norm(sub_img-img/255)
            # replace original image by subtractive image and store the new hash
            print(f'[INFO] Saving Subtracted Image after {i} iterations')
            save_img(f'../{filename}_new.{filetype}', sub_img)
            break
    print(f'\nThe distortion to the original image is {dist:.2f} units')

def simba_attack_batch(folder_path, batch):
    img_paths = load_img_paths(folder_path)
    resize_imgs(img_paths, batch=batch)
    imgs = load_img(img_paths, batch=batch)
    add_imgs, sub_imgs = copy.deepcopy(imgs), copy.deepcopy(imgs)
    init_hashes = compute_hash(img_paths, batch=batch)
    epsilon, i = 0.9, 0
    stepsize = int(epsilon*255.0)
    # print(init_hashes)
    while True:
        break



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", help="Path to the image for the single image attack")
    ap.add_argument("-f", "--folder_path", help="Path to the directory containing images for the batch image attack")
    args = vars(ap.parse_args())

    if args['img_path'] is not None:
        img_path = args['img_path']
        batch = False
    if args['folder_path'] is not None:
        folder_path = args['folder_path']
        batch = True

    if not batch:
        simba_attack_image(img_path)
    else:
        simba_attack_batch(folder_path, batch)