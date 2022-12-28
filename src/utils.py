from PIL import Image
import numpy as np
import subprocess
import cv2

def compute_hash(image_path, batch=False):
    hashing_file_name = './nhcalc'
    if batch:
        image_path.insert(0, hashing_file_name)
        output = subprocess.check_output(image_path)
        hashes = output.strip().split()
        hashes = hashes[1::3]
        return list(map(lambda x: int(x,16), hashes))
    else:
        output = subprocess.check_output([hashing_file_name, image_path])
        hash = output.strip().split()
        return int(hash[1], 16)

def sample_pixel(img):
    (Y, X) = img.shape[0], img.shape[1]
    (Y, X) = (int(Y*np.random.random(1)), int(X*np.random.random(1)))
    pixel = img[Y][X]
    return (pixel, Y, X)
 
def save_img(save_path, img):
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)

def load_img(img_path):
    return np.array(Image.open(img_path)).astype(np.float32)