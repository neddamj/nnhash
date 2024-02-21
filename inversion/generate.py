import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Hash2ImageModel
from data import Hash2ImgDataset
from hash import compute_hash, hamming_distance

import matplotlib.pyplot as plt

# Constants for training
NUM_EPOCHS = 35
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

torch.manual_seed(1337)

# Create the dataset and data loader for training
try:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
except:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
dataset = Hash2ImgDataset(image_paths='./_data/val/images', hash_paths='./_data/val/hashes.pkl', transforms=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Load the saved model
model = Hash2ImageModel()
checkpoint = torch.load('/Users/neddamj/Documents/BU/Research/2022PhotoDNA/nnhash/inversion/saved_models/2024-02-21_13:31:44%_saved_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)

model.eval()
avg_hamm_dist = []
for i, (hash, image) in enumerate(loader):
    if i == 15:
        break
    hash, image = hash.to(DEVICE), image.to(DEVICE)
    with torch.no_grad():
        pred_img = model(hash)   
    pred_img = pred_img.to(torch.device('cpu'))
    pred_img = pred_img.squeeze(0).detach().permute(1, 2, 0)
    image = image.to(torch.device('cpu'))
    image = image.squeeze(0).detach().permute(1, 2, 0)
    # Compute the hamming distance between the predicted image
    # and the original
    pred_hash = compute_hash(pred_img.permute(2, 0, 1))
    true_hash = compute_hash(image.permute(2, 0, 1))
    hamm_dist = hamming_distance(true_hash, pred_hash)
    avg_hamm_dist.append(hamm_dist)

    fig = plt.figure(figsize=(5, 5))
    rows, cols = 1, 2
    fig.add_subplot(rows, cols, 1)
    plt.imshow(pred_img)
    plt.title('Prediction')
    fig.add_subplot(rows, cols, 2)
    plt.imshow(image)
    plt.title('Ground Truth')
    plt.show()

print(f'The average hamming distance between the hashes of the generations and the originals is {sum(avg_hamm_dist)/len(avg_hamm_dist)}')