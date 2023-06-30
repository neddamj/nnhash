from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import subprocess
import shutil
import glob
import os

## Adversarial Helper Functions
def distance(x, y, type='l2'):
    if type == 'l2':
        return np.linalg.norm(x/255.0 - y/255.0)
    elif type == 'linf':
        return np.max(abs(x/255.0 - y/255.0))
    elif type == 'ssim':
        return ssim(x, y, channel_axis=-1)
    elif type == 'hamming':
        # x and y in this case refer to the hash values of the 2 images
        # rather than the images themselves
        return bin(x^y).count('1')

def compute_hash(image_path, batch=False, hash_file_path='../nhcalc'):
    hashing_file_name = hash_file_path
    if batch:
        image_path.insert(0, hashing_file_name)
        output = subprocess.check_output(image_path)
        image_path.pop(0)
        hashes = output.strip().split()
        hashes = hashes[1::3]
        return np.array(list(map(lambda x: int(x,16), hashes)))
    else:
        # Handle the input when an image is supplied instead of the path
        if type(image_path) == np.ndarray:
            try:
                path = '../images/hash.jpeg'
                save_img(path, image_path)
            except:
                path = '../../images/hash.jpeg'
                save_img(path, image_path)
            image_path = path
        output = subprocess.check_output([hashing_file_name, image_path])
        hash = output.strip().split()
        return int(hash[1], 16)

## Data Helper Functions
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

def move_data_to_temp_ram(path, ram_size_mb=1, batch=False):
    src_dir = path
    dst_dir = '/Volumes/TempRAM/'
    if not os.path.exists(dst_dir):
        os.system(f'diskutil erasevolume HFS+ "TempRAM" `hdiutil attach -nomount ram://{ram_size_mb*2048}`')
        print('[INFO] RAM Disk Created...')
    else:
        print('[INFO] RAM Disk Already Exists...')
        paths = load_img_paths(dst_dir)
        for path in paths:
            os.remove(path)
        print('[INFO] RAM Disk Cleared...')
    for file_name in os.listdir(src_dir):
        source = src_dir + file_name
        destination = dst_dir + file_name
        if os.path.isfile(source):
            shutil.copy(source, destination)
    print('[INFO] Files moved to temp RAM...')
    return dst_dir