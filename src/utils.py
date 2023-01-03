from PIL import Image
import numpy as np
import subprocess
import glob
import cv2

def resize_imgs(image_paths, new_size=(512, 512), batch=False):
    if batch:
        for path in image_paths:
            img = Image.open(path)
            img = img.resize(new_size)
            img.save(path)
    else:
        img = Image.open(image_paths)
        img = img.resize(new_size)
        img.save(image_paths)

def compute_hash(image_path, batch=False):
    hashing_file_name = './nhcalc'
    if batch:
        image_path.insert(0, hashing_file_name)
        output = subprocess.check_output(image_path)
        image_path.pop(0)
        hashes = output.strip().split()
        hashes = hashes[1::3]
        return np.array(list(map(lambda x: int(x,16), hashes)))
    else:
        output = subprocess.check_output([hashing_file_name, image_path])
        hash = output.strip().split()
        return int(hash[1], 16)

def sample_pixel(img, prev_samples, batch=False):
    def sample(img, batch):
        if not batch:
            (H, W) = img.shape[0], img.shape[1]
            (Y, X) = (int(H*np.random.random(1)), int(W*np.random.random(1)))
            pixel = img[Y][X]
        else:   
            (Y, X) = (int(512*np.random.random(1)), int(512*np.random.random(1)))
            pixel = list(map(lambda x: x[Y][X], img))
        return (pixel, Y, X)
    # Sample without replacement
    (pixel, Y, X) = sample(img, batch=batch)
    while (Y, X) in prev_samples:
        (pixel, Y, X) = sample(img, batch=batch)
    prev_samples.append((Y,X))
    return (pixel, Y, X), prev_samples

def load_img_paths(img_folder):
    if img_folder[-1] == '/':
        return glob.glob(f'{img_folder}*')
    return glob.glob(f'{img_folder}/*')
 
def save_img(save_path, img):
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)

def load_img(img_path, batch=False):
    if batch:
        return np.array(list(map(lambda x: np.array(Image.open(x)), img_path)))
    else:
        return np.array(Image.open(img_path))