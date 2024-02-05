import torch
from torchvision import datasets
from torch.utils.data import Dataset, random_split
from hash import compute_hash, hash2tensor
from PIL import Image
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

def save_img(save_path, img):
    img.save(save_path)

if __name__ == '__main__':
    # Load the train and val splits
    dataset = 'celeba'
    if dataset == 'celeba':
        dataset = datasets.CelebA(root='./', split='train', target_type='identity', download=False)
    else:
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    train_len, val_len = 10000, 1000
    train_data, val_data, _ = random_split(dataset, [train_len, val_len, len(dataset)-11000])

    def save_data(split):
        data = train_data if split == 'train' else val_data
        hashes = []
        for i, sample in enumerate(data):
            save_path = f'./_data/{split}/images/{i+1}.jpeg'
            img = sample[0]
            img = img.resize((64, 64))
            save_img(save_path, img)
            hashes.append(compute_hash(save_path))
            if (i+1) % 500 == 0:
                print(f'{i} image-hash pairs saved')

        # save the hashes for each image
        hash_file_path = f'./{split}_data/hashes.pkl'
        with open(hash_file_path, 'wb') as f:
            pickle.dump(hashes, f)

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
        save_data(split)
        print(f'[INFO] Complete ...')