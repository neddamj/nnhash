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
    def __init__(self, image_paths, hash_paths, transforms=None):
        self.transforms = transforms
        self.image_paths = image_paths
        with open(hash_paths, 'rb') as f:
            self.hashes = pickle.load(f)

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        # Get the image at a specified index and its hash tensor
        image_path = f'{self.image_paths}/{index+1}.jpeg'
        image = Image.open(image_path)
        hash = hash2tensor(self.hashes[index])

        # Apply specified transforms
        if self.transforms:
            image = self.transforms(image)
        
        return hash, image
    
def save_data(split, dataset):
    data = train_data if split == 'train' else val_data
    hashes = []
    for i, sample in enumerate(data):
        save_path = f'./_data/{split}/images/{i+1}.jpeg'
        img = sample[0]
        img = img.resize((64, 64))
        save_img(save_path, img)
        hashes.append(compute_hash(save_path))
        if (i+1) % 500 == 0:
            print(f'{i+1} image-hash pairs saved')

    # save the hashes for each image
    hash_file_path = f'./_data/{split}/hashes.pkl'
    with open(hash_file_path, 'wb') as f:
        pickle.dump(hashes, f)

def save_img(save_path, img):
    img.save(save_path)

if __name__ == '__main__':
    '''
    This will load the data into a pytorch dataset and then save the image/hash pairs to
    memory directly.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='The name of the dataset you want to use in lower case', type=str)
    parser.add_argument('--train_len', help='The amount of datapoints to be used for training', default=10000, type=int)
    parser.add_argument('--val_len', help='The amount of datapoints to be used for testing', default=100, type=int)
    parser.add_argument('--skip', help='If no data loading is necessary then skip this step. Set to true if you need to change datasets before training/inference.', 
                        default=0, type=int)
    args = parser.parse_args()
    
    # Load the train and val splits
    if not args.skip:
        data = args.dataset
        if data == 'celeba':
            dataset = datasets.CelebA(root='./', split='train', target_type='identity', download=False)
        elif data == 'mnist':
            dataset = datasets.MNIST(root='./mnist', train=True, download=True, transform=None)
        elif data == 'stl10':
            dataset = datasets.STL10(root='./stl10', split='train', download=True)
        else:
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        train_len, val_len = args.train_len, args.val_len
        train_data, val_data, _ = random_split(dataset, [train_len, val_len, len(dataset)-(train_len+val_len)])
        print(dataset)
        splits = ['train', 'val']
        for split in splits:
            # Create the dirs if they dont exist already
            dir_path = f'./_data'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            sub_dir_path = f'{dir_path}/{split}'
            if not os.path.exists(sub_dir_path):
                os.makedirs(sub_dir_path)
            if not os.path.exists(f'{sub_dir_path}/images'):
                os.makedirs(f'{sub_dir_path}/images')
            # Save the data
            print(f'[INFO] Saving {split} data ...')
            save_data(split, dataset)
            print(f'[INFO] Complete ...')