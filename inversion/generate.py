import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Hash2ImageModel
from data import Hash2ImgDataset
from hash import compute_hash, hamming_distance

from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Constants for training
NUM_EPOCHS = 35
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

torch.manual_seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('-r','--rgb', help='Are the images in the dataset rgb or greyscale', default=0, type=int)
parser.add_argument('-d', '--display', help='Display the generated images or not', default=0, type=int)
args = parser.parse_args()

# Using color images or not
rgb = args.rgb


# Create the dataset and data loader for training
if rgb:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
dataset = Hash2ImgDataset(image_paths='./_data/val/images', hash_paths='./_data/val/hashes.pkl', transforms=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Load the saved model
model = Hash2ImageModel(rgb)
checkpoint = torch.load('/Users/neddamj/Documents/BU/Research/2022PhotoDNA/nnhash/inversion/saved_models/2024-02-21_13:31:44%_mnist_saved_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)

model.eval()
avg_hamm_dist, avg_l2_dist, avg_ssim = [], [], []
for i, (hash, image) in enumerate(loader):
    if i == 100:
        break
    hash, image = hash.to(DEVICE), image.to(DEVICE)
    with torch.no_grad():
        pred_img = model(hash)   
    pred_img = pred_img.to(torch.device('cpu'))
    pred_img = pred_img.squeeze(0).detach().permute(1, 2, 0)
    image = image.to(torch.device('cpu'))
    image = image.squeeze(0).detach().permute(1, 2, 0)
    # Find per-pixel L2 distance between images. Images are already normalized to the
    # range [0, 1] so no need to normalize while calculating per-pixel norm
    if not rgb:
        l2_dist = np.linalg.norm(image - pred_img)/np.sqrt(64*64)
    else:
        l2_dist = np.linalg.norm(image - pred_img)/np.sqrt(64*64*3)
    avg_ssim.append(ssim(np.array(image), np.array(pred_img), channel_axis=-1, data_range=1))
    avg_l2_dist.append(l2_dist)
    # Compute the hamming distance between the predicted image
    # and the original
    pred_hash = compute_hash(pred_img.permute(2, 0, 1))
    true_hash = compute_hash(image.permute(2, 0, 1))
    hamm_dist = hamming_distance(true_hash, pred_hash)
    avg_hamm_dist.append(hamm_dist)

    if args.display:
        fig = plt.figure(figsize=(5, 5))
        rows, cols = 1, 2
        fig.add_subplot(rows, cols, 1)
        plt.imshow(pred_img)
        plt.title('Prediction')
        fig.add_subplot(rows, cols, 2)
        plt.imshow(image)
        plt.title('Ground Truth')
        plt.show()

print(f'The average l2 distance is {np.array(avg_l2_dist).mean()} +- {np.std(np.array(avg_l2_dist))}')
print(f'The average SSIM is {np.array(avg_ssim).mean()} +- {np.std(np.array(avg_ssim))}')
print(f'The hamming distance between the images is {np.array(avg_hamm_dist).mean()}')