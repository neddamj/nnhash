'''
    Usage: python3 simba_attack.py
'''

from utils import compute_hash, sample_pixel, save_img, resize_imgs
from PIL import Image
import numpy as np
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
    resize_imgs(img_path)
    # Initialize images
    img = np.array(Image.open(img_path)).astype(np.float32)
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
            dist = np.linalg.norm(sub_img-img)
            # replace original image by subtractive image and store the new hash
            print(f'[INFO] Saving Subtracted Image after {i} iterations')
            save_img(f'../{filename}_new.{filetype}', sub_img)
            break
    print(f'\nThe distortion to the original image is {dist:.2f} units')

def simba_attack_batch():
    pass

if __name__ == "__main__":
    img_path = '../images/02.jpeg'
    batch = False
    if not batch:
        simba_attack_image(img_path)
    else:
        simba_attack_batch()