import torch
from torchvision import datasets
from torch.utils.data import Dataset, random_split
from hash import compute_hash, hash2tensor
from PIL import Image
import numpy as np
import argparse
import pickle
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
    
def save_data(split, hash_func):
    data = train_data if split == 'train' else val_data
    hashes = []
    for i, sample in enumerate(data):
        save_path = f'./_data/{split}/images/{i+1}.jpeg'
        img = sample[0]
        img = img.resize((64, 64))
        save_img(save_path, img)
        hashes.append(compute_hash(save_path, hash_func=hash_func))
        if (i+1) % 500 == 0:
            print(f'{i+1} image-hash pairs saved')

    # save the hashes for each image
    hash_file_path = f'./_data/{split}/hashes.pkl'
    with open(hash_file_path, 'wb') as f:
        pickle.dump(hashes, f)

def save_img(save_path, img):
    img.save(save_path)

def int_to_binary(tensor):
    # Ensure the tensor is of the correct shape and type
    if tensor.shape != (1, 144) or not torch.all((0 <= tensor) & (tensor <= 255)):
        raise ValueError("Input tensor must be of shape [1, 144] with values in the range [0, 255].")
    
    # Convert the tensor to uint8 type
    tensor = tensor.to(torch.uint8)
    
    # Convert to binary representation using bit shifts and bitwise AND
    binary_tensor = ((tensor.unsqueeze(-1) >> torch.arange(8, dtype=torch.uint8)) & 1).flatten().view(1, -1)
    
    return binary_tensor

def binary_to_int(binary_tensor):
    # Ensure the tensor is of the correct shape
    if binary_tensor.shape != (1, 1152) or not torch.all((binary_tensor == 0) | (binary_tensor == 1)):
        raise ValueError("Input tensor must be of shape [1, 1152] with binary values {0, 1}.")
    
    # Reshape the binary tensor to have 8 columns
    binary_tensor = binary_tensor.view(-1, 8)
    
    # Convert each 8-bit segment back to its integer form
    powers_of_two = 2 ** torch.arange(8, dtype=torch.uint8)
    int_tensor = (binary_tensor * powers_of_two).sum(dim=1).view(1, -1)
    
    return int_tensor

def perturb_hash(hash, p=0.1, hash_func='pdq'):
    if p == 0:
        return hash
    if hash_func == 'photodna':
        hash = int_to_binary(hash)
        mask_indices = torch.multinomial(hash.float(), int(hash.numel()*p), replacement=False)
        mask = torch.ones_like(hash).int(); mask[0][mask_indices] = 0
        print(mask, mask_indices)
        perturbed_hash = torch.tensor([[int(hash[0][i].item()) if mask[0][i] else not int(hash[0][i].item()) for i in range(1152)]]).int()
        perturbed_hash = binary_to_int(perturbed_hash)
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