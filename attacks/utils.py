from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import subprocess
import shutil
import pdqhash
import random
import torch
import glob
import os

from ctypes import cast
from ctypes import cdll
from ctypes import c_int
from ctypes import c_ubyte
from ctypes import POINTER
from ctypes import c_char_p

## Adversarial Helper Functions
def distance(x, y, type='l2', hash_func='neuralhash'):
    if type == 'l2':
        return np.linalg.norm(x/255.0 - y/255.0)
    elif type == 'linf':
        return np.max(abs(x/255.0 - y/255.0))
    elif type == 'ssim':
        return ssim(x, y, channel_axis=-1)
    elif type == 'hamming':
        # x and y in this case refer to the hash values of the 2 images
        # rather than the images themselves
        if hash_func == 'neuralhash':
            hamming_dist = bin(x^y).count('1')
        elif hash_func == 'pdq':
            hash_x = int(''.join([str(num) for num in x]), 2)
            hash_y = int(''.join([str(num) for num in y]), 2)
            # Calculate hammng dist
            hamming_dist = bin(hash_x^hash_y).count('1')
        return hamming_dist

def compute_hash(image_path, batch=False, hash_file_path='../nhcalc', hash_func='neuralhash'):
    hashing_file_name = hash_file_path
    if hash_func == 'neuralhash':
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
                    path = os.path.sep.join(['..', 'images', 'hash.bmp'])
                    save_img(path, image_path)
                except:
                    path = os.path.sep.join(['..', '..', 'images', 'hash.bmp'])
                    save_img(path, image_path)
                    image_path = path
            output = subprocess.check_output([hashing_file_name, image_path])
            hash = output.strip().split()
            hash = int(hash[1], 16)
            return hash
    elif hash_func == 'pdq':
        if type(image_path) == np.ndarray:
            try:
                path = os.path.sep.join(['..', 'images', 'hash.bmp'])
                save_img(path, image_path)
            except:
                path = os.path.sep.join(['..', '..', 'images', 'hash.bmp'])
                save_img(path, image_path)
            image_path = path
        image = load_img(image_path)
        hash, _ = pdqhash.compute(image)
        return hash
    else: # photodna
        if type(image_path) == np.ndarray:
            try:
                path = os.path.sep.join(['..', 'images', 'hash.bmp'])
                save_img(path, image_path)
            except:
                path = os.path.sep.join(['..', '..', 'images', 'hash.bmp'])
                save_img(path, image_path)
            image_path = path
            hash_val = generatePhotodnaHash(image_path)
            return hash_val
    
def generatePhotodnaHash(imagePath):
    outputFolder = os.getcwd()
    libName = os.path.join('..', 'PhotoDNAx64.dll')
    #workerId = multiprocessing.current_process().name
    imageFile = Image.open(imagePath, 'r')
    if imageFile.mode != 'RGB':
        imageFile = imageFile.convert(mode = 'RGB')
    libPhotoDNA = cdll.LoadLibrary(os.path.join(outputFolder, libName))

    ComputeRobustHash = libPhotoDNA.ComputeRobustHash
    ComputeRobustHash.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
    ComputeRobustHash.restype = c_ubyte

    hashByteArray = (c_ubyte * 144)()
    ComputeRobustHash(c_char_p(imageFile.tobytes()), imageFile.width, imageFile.height, 0, hashByteArray, 0)

    hashPtr = cast(hashByteArray, POINTER(c_ubyte))
    hashList = [str(hashPtr[i]) for i in range(144)]
    hashString = ','.join([i for i in hashList])
    hashList = hashString.split(',')
    return hashList

def hash2tensor(hash, hash_func='pdq'):
    if hash_func == 'neuralhash':
        hash = bin(hash)[2:]
        hash_tensor = torch.tensor([float(num) for num in hash])
        if hash_tensor.shape != torch.Size([128]):
            num_zeros = 128 - hash_tensor.shape[0]
            hash_tensor = torch.cat((torch.tensor([0 for _ in range(num_zeros)]), hash_tensor), dim=0)
    elif hash_func == 'pdq':
        hash_tensor = torch.tensor(hash.copy())
    else: # photodna
        hash_tensor = torch.tensor([float(num) for num in hash])
    return hash_tensor

def tensor2hash(tensor: torch.Tensor, hash_func='pdq'):
    if hash_func == 'neuralhash':
        raise NotImplementedError
    elif hash_func == 'pdq':
        hash_val = tensor.numpy()
    else: # photodna
        raise NotImplementedError
    return hash_val

def perturb_hash_tensor(hash, p=0.1, hash_func='pdq'):
    if p == 0.0:
        return hash
    if hash_func == 'photodna':
        def bitwise_flip(tensor, p=0.1):
            flat_tensor = tensor.flatten().to(torch.uint8)
            # Calculate the number of elements to flip
            num_elements_to_flip = int(p * flat_tensor.numel())
            # Randomly select indices to flip
            indices_to_flip = random.sample(range(flat_tensor.numel()), num_elements_to_flip)
            # Perform bitwise flip on the selected elements
            for idx in indices_to_flip:
                flat_tensor[idx] = flat_tensor[idx] ^ 0xFF
            return flat_tensor.reshape(tensor.shape)
        perturbed_hash = bitwise_flip(hash, p=p)
    else:
        mask_indices = torch.multinomial(hash.float(), int(hash.numel()*p), replacement=False)
        mask = torch.ones_like(hash).int()
        if hash_func == 'pdq':
            mask[mask_indices] = 0
            perturbed_hash = torch.tensor([int(hash[i].item()) if mask[i] else (not int(hash[i].item())) for i in range(256)]).int()
        else: # neuralhash
            mask[mask_indices] = 0
            perturbed_hash = torch.tensor([int(hash[i].item()) if mask[i] else (not int(hash[i].item())) for i in range(128)]).int()
    return perturbed_hash

def perturb_hash(hash_val, p=0.1, hash_func='pdq'):
    # Convert the hash to a tensor
    h_tensor = hash2tensor(hash_val, hash_func=hash_func)
    # Perturb the hash tensor
    h_perturbed = perturb_hash_tensor(h_tensor, p=p, hash_func=hash_func)
    # Convert the tensor back to a hash
    h_val = tensor2hash(h_perturbed, hash_func=hash_func)
    return h_val

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
    if img_folder[-1] == os.path.sep:
        return glob.glob(os.path.join(img_folder, '*')) 
    return glob.glob(os.path.sep.join([img_folder, '*'])) 
 
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