from PIL import Image
import numpy as np
import subprocess
import shutil
import glob
import os

def move_data_to_temp_ram(path, ram_size_mb=1, batch=False):
    src_dir = path
    dst_dir = '/Volumes/TempRAM/'
    if not os.path.exists(dst_dir):
        os.system(f'diskutil erasevolume HFS+ "TempRAM" `hdiutil attach -nomount ram://{ram_size_mb*2048}`')
        print('[INFO] RAM Disk Created...')
    else:
        print('[INFO] RAM Disk Already Exists...')
    for file_name in os.listdir(src_dir):
        source = src_dir + file_name
        destination = dst_dir + file_name
        if os.path.isfile(source):
            shutil.copy(source, destination)
    print('[INFO] Files moved to temp RAM...')
    return dst_dir

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

def sample_pixel(img, batch=False):
    if not batch:
        (H, W) = img.shape[0], img.shape[1]
        (Y, X) = (int(H*np.random.random(1)), int(W*np.random.random(1)))
        pixel = img[Y][X]
    else:   
        (Y, X) = (int(512*np.random.random(1)), int(512*np.random.random(1)))
        pixel = list(map(lambda x: x[Y][X], img))
    return (pixel, Y, X)


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