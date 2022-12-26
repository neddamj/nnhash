from utils import compute_hash, sample_pixel, save_img, load_img
import numpy as np
import torch
import cv2

def get_hash_of_preturbed_img(pixel, H, W, path, img, stepsize):
    _, _, filename, filetype = path.split('.')
    add_img, sub_img = np.copy(img), np.copy(img)
    # Add value to the pixel and get the hash 
    add_img[H][W] = pixel + stepsize
    save_img(f'../{filename}_add.{filetype}', add_img)
    addititve_hash = compute_hash(f'../{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    sub_img[H][W] = pixel - stepsize
    save_img(f'../{filename}_sub.{filetype}', sub_img)
    subtractive_hash = compute_hash(f'../{filename}_sub.{filetype}')
    return (add_img, addititve_hash, sub_img, subtractive_hash)

if __name__ == "__main__":
    img_path = '../images/c1.bmp'
    img = cv2.imread(img_path).astype(np.float32)
    init_hash = compute_hash(img_path)
    stepsize = 1.0
    for i in range(10000):
        (pixel, Y, X) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_preturbed_img(pixel, Y, X, img_path, img, stepsize)
        print(f'Iteration: {i+1}\tAdd Hash: {hex(additive_hash)}\tSub Hash: {hex(subtractive_hash)}')
        if abs(init_hash-additive_hash) > abs(init_hash-subtractive_hash):
            # replace original image by additive image
            print('[INFO] Saving Added Image')
            save_img(img_path, add_img)
        elif abs(init_hash-additive_hash) < abs(init_hash-subtractive_hash):
            # replace original image by subtractive image
            print('[INFO] Saving Subtracted Image')
            save_img(img_path, sub_img)
        