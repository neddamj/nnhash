from PIL import Image
import torchvision.transforms as T
import numpy as np
import subprocess
import torch

def compute_hash(image_path, hash_file_path='./nhcalc'):
    hashing_file_name = hash_file_path
    # Handle the input when an array/tensor is supplied instead of the path
    if type(image_path) == np.ndarray:
        path = './imgs/hash.bmp'
        img = Image.fromarray(image_path)
        img.save(path)
        image_path = path
    elif type(image_path) == torch.Tensor:
        path = './imgs/hash.bmp'
        transform = T.ToPILImage()
        img = transform(image_path)
        img.save(path)
        image_path = path
    output = subprocess.check_output([hashing_file_name, image_path])
    hash = output.strip().split()
    return int(hash[1], 16)

def hash2tensor(hash):
    hash = bin(hash)[2:]
    hash_tensor = torch.tensor([float(num) for num in hash])
    if hash_tensor.shape != torch.Size([128]):
        num_zeros = 128 - hash_tensor.shape[0]
        hash_tensor = torch.cat((torch.tensor([0 for _ in range(num_zeros)]), hash_tensor), dim=0)
    return hash_tensor

def hamming_distance(hash_1, hash_2):
    return bin(hash_1^hash_2).count('1')

if __name__ == '__main__':
    img_path = './imgs/hash.bmp'
    hash = compute_hash(img_path)
    hash_tensor = hash2tensor(hash)
    print(hash_tensor.shape, hash_tensor)