from utils import compute_hash, sample_pixel, save_img
from PIL import Image
import numpy as np
import copy
import cv2

def get_hash_of_preturbed_imgs(pixel, H, W, path, add_img, sub_img, stepsize):
    _, _, filename, filetype = path.split('.')
    # Add value to the pixel and get the hash 
    if pixel + stepsize <= 255.0:
        add_img[H][W] = pixel + stepsize
    else:
        add_img[H][W] == 255.0
    save_img(f'../{filename}_add.{filetype}', add_img)
    add_hash = compute_hash(f'../{filename}_add.{filetype}')
    # Subtract value from the pixel and get the hash 
    if pixel - stepsize >= 0.0:
        sub_img[H][W] = pixel - stepsize
    else:
        sub_img[H][W] = 0.0
    save_img(f'../{filename}_sub.{filetype}', sub_img)
    sub_hash = compute_hash(f'../{filename}_sub.{filetype}')
    return (add_img, add_hash, sub_img, sub_hash)

if __name__ == "__main__":
    img_path = '../images/01.jpg'
    _, _, filename, filetype = img_path.split('.')
    # Initialize images
    img = np.array(Image.open(img_path)).astype(np.float32)
    add_img, sub_img = copy.deepcopy(img), copy.deepcopy(img)
    init_hash = compute_hash(img_path)
    stepsize, counter, i = 200.0, 0, 0
    # Store the initial hash
    hashes = []
    hashes.append(init_hash)
    while True:
        i += 1
        (pixel, Y, X) = sample_pixel(img)
        (add_img, additive_hash, sub_img, subtractive_hash) = get_hash_of_preturbed_imgs(pixel, Y, X, img_path, add_img, sub_img, stepsize)
        print(f'Iteration: {i} \tAdd Hash: {hex(additive_hash)} \tSub Hash: {hex(subtractive_hash)}')
        if abs(init_hash-additive_hash) > abs(init_hash-subtractive_hash):
            # replace original image by additive image and store the new hash
            counter += 1
            hashes.append(additive_hash)
            print(f'[INFO] Saving Added Image after {i} iterations')
            save_img(f'../{filename}_new.{filetype}', add_img)
            break
        elif abs(init_hash-additive_hash) < abs(init_hash-subtractive_hash):
            # replace original image by subtractive image and store the new hash
            counter += 1
            hashes.append(subtractive_hash)
            print(f'[INFO] Saving Subtracted Image after {i} iterations')
            save_img(f'../{filename}_new.{filetype}', sub_img)
            break
    print(f'The hash was changed {counter} times')