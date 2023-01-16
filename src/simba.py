from utils import save_img, compute_hash
import numpy as np

def get_hash_of_imgs(pixels, H, W, C, path, add_img, sub_img, stepsize):
    filename, filetype = path.split('.') 
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