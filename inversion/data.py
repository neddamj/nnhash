import torch
from torchvision import datasets
from torch.utils.data import Dataset, random_split
from hash import compute_hash, hash2tensor
from PIL import Image
import numpy as np
import argparse
import pickle
import random
import os

class Hash2ImgDataset(Dataset):
    def __init__(self, image_paths, hash_paths, hash_func='pdq', transforms=None, perturbation=0):
        self.transforms = transforms
        self.image_paths = image_paths
        self.hash_func = hash_func
        self.perturbation = perturbation
        with open(hash_paths, 'rb') as f:
            self.hashes = pickle.load(f)

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        # Get the image at a specified index and its hash tensor
        image_path = f'{self.image_paths}/{index+1}.jpeg'
        image = Image.open(image_path)
        hash = hash2tensor(self.hashes[index], hash_func=self.hash_func)
        hash = perturb_hash(hash, p=self.perturbation, hash_func=self.hash_func)

        # Apply specified transforms
        if self.transforms:
            image = self.transforms(image)

        return hash, image

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

def perturb_hash(hash, p=0.1, hash_func='pdq'):
    if p == 0:
        return hash
    if hash_func == 'photodna':
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

if __name__ == '__main__':
    '''
    This will load the data into a pytorch dataset and then save the image/hash pairs to
    memory directly.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='The name of the dataset you want to use in lower case', type=str)
    parser.add_argument('--train_len', help='The amount of datapoints to be used for training', default=10000, type=int)
    parser.add_argument('--val_len', help='The amount of datapoints to be used for testing', default=100, type=int)
    parser.add_argument('--hash_func', help='Hash function that you want to invert', default='pdq', type=str)
    parser.add_argument('--skip', help='If no data loading is necessary then skip this step. Set to true if you need to change datasets before training/inference.', 
                        default=0, type=int)
    args = parser.parse_args()
    
    # Load the train and val splits
    if not args.skip:
        root_path = os.path.sep.join(['.', 'data'])
        data = args.dataset
        if data == 'celeba':
            dataset = datasets.CelebA(root=root_path, split='train', target_type='identity', transform=None, download=True)
        elif data == 'mnist':
            dataset = datasets.MNIST(root=root_path, train=True, download=True, transform=None)
        elif data == 'stl10':
            dataset = datasets.STL10(root=root_path, split='train', download=True)
        elif data == 'fashion':
            dataset = datasets.FashionMNIST(root=root_path, train=True, download=False)
        else:
            dataset = datasets.CIFAR10(root=root_path, train=True, download=True, transform=None)
        train_len, val_len = args.train_len, args.val_len
        train_data, val_data, _ = random_split(dataset, [train_len, val_len, len(dataset)-(train_len+val_len)])
        print(dataset)
        splits = ['train', 'val']
        for split in splits:
            # Create the dirs if they dont exist already
            dir_path = os.path.sep.join(['.', '_data'])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            sub_dir_path = os.path.sep.join([dir_path, split])  #f'{dir_path}/{split}'
            if not os.path.exists(sub_dir_path):
                os.makedirs(sub_dir_path)
            if not os.path.exists(os.path.sep.join([sub_dir_path, 'images'])):
                os.makedirs(os.path.sep.join([sub_dir_path, 'images']))
            # Save the data
            print(f'[INFO] Saving {split} data ...')
            save_data(split, hash_func=args.hash_func)
            print(f'[INFO] Complete ...')