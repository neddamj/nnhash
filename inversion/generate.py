import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Hash2ImageModel, STL10Hash2ImageModel
from data import Hash2ImgDataset

from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import lpips
import os

BATCH_SIZE = 1
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"

torch.manual_seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('-r','--rgb', help='Are the images in the dataset rgb or greyscale', default=0, type=int)
parser.add_argument('-d', '--display', help='Display the generated images or not', default=0, type=int)
parser.add_argument('-f','--hash_func', help='Hash function to be inverted', required=True, type=str)
parser.add_argument('-d','--dataset', help='Dataset to be inverted', required=True, type=str)
parser.add_argument('-u', '--perturb', help='Magnitude of the perturbation', required=True, type=float)
parser.add_argument('-p','--path', help='Path to the saved torch model to be used', required=True, type=str)
args = parser.parse_args()

# Using color images or not
rgb = args.rgb 

# Create the dataset and data loader for training
if rgb:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.dataset == 'stl10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize((0.5), (0.5))
        ])
        rgb = False
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
dataset = Hash2ImgDataset(image_paths=os.path.sep.join(['.', '_data', 'val', 'images']), 
                          hash_paths=os.path.sep.join(['.', '_data', 'val', 'hashes.pkl']), 
                          hash_func=args.hash_func, 
                          transforms=transform,
                          perturbation=args.perturb)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Load the saved model 
if args.dataset == 'stl10':
    model = STL10Hash2ImageModel(rgb=rgb, hash_func=args.hash_func)
else:
    model = Hash2ImageModel(rgb=rgb, hash_func=args.hash_func)
checkpoint = torch.load(args.path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Load the learned perceptual similarity metric
perceptual_sim = lpips.LPIPS(net='vgg')

avg_hamm_dist, avg_l2_dist, avg_ssim, avg_lpips = [], [], [], []
for i, (hash, image) in enumerate(loader):
    if i == 100:
        break
    hash, image = hash.to(DEVICE), image.to(DEVICE)
    with torch.no_grad():
        pred_img = model(hash)   
    pred_img = pred_img.to(torch.device('cpu'))
    pred_img = pred_img.squeeze(0).detach().permute(1, 2, 0)
    pred_img = (pred_img + 1)/2
    image = image.to(torch.device('cpu'))
    image = image.squeeze(0).detach().permute(1, 2, 0)
    image = (image + 1)/2
    
    # Find per-pixel L2 distance between images. Images are already normalized to the
    # range [0, 1] so no need to normalize while calculating per-pixel norm
    if not rgb:
        l2_dist = np.linalg.norm(image - pred_img)/np.sqrt(64*64)
    else:
        l2_dist = np.linalg.norm(image - pred_img)/np.sqrt(64*64*3)
    print(f'L2 Distance: {l2_dist}')
    avg_l2_dist.append(l2_dist)
    # Find the SSIM between the original and geneerated images
    ssim_score = ssim(np.array(image), np.array(pred_img), channel_axis=-1, data_range=1)
    print(f'SSIM Score: {ssim_score}')
    avg_ssim.append(ssim_score)
    # Calculate the perceptual similarity
    image, pred_img = image.permute(2, 0, 1), pred_img.permute(2, 0, 1)
    d = perceptual_sim(image, pred_img, normalize=True)
    print(f'LPIPS Score: {d.detach().numpy()}')
    avg_lpips.append(d.detach().numpy())
    
    image, pred_img = image.permute(1, 2, 0), pred_img.permute(1, 2, 0)

    if args.display:
        fig = plt.figure(figsize=(5, 5))
        rows, cols = 1, 2
        fig.add_subplot(rows, cols, 1)
        plt.imshow(pred_img)
        plt.title('Prediction')
        plt.axis('off')
        fig.add_subplot(rows, cols, 2)
        plt.imshow(image)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.show()

print(f'The average l2 distance is {np.array(avg_l2_dist).mean()} +- {np.std(np.array(avg_l2_dist))} units')
print(f'The average SSIM is {np.array(avg_ssim).mean()} +- {np.std(np.array(avg_ssim))} units')
print(f'The average LPIPS distance between the images is {np.array(avg_lpips).mean()} +- {np.std(np.array(avg_lpips))} units')