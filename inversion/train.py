import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from hash import hash2tensor
from model import Hash2ImageModel
from data import Hash2ImgDataset

from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the dataset being used to train the model', required=True, type=str)
    parser.add_argument('--rgb', help='Are the images in the dataset rgb or greyscale', default=0, type=int)
    parser.add_argument('--epochs', help='Number of Epochs to train for', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size to be used in training', default=32, type=int)
    parser.add_argument('--hash_func', help='Hash function that you want to invert', default='pdq', type=str)
    parser.add_argument('perturbation', help='Magnitude of perturbation to be applied to the hashes', default=0.0, type=float)
    parser.add_argument('--learning_rate', help='Learning Rate to be used during training', default=5e-4, type=float)
    args = parser.parse_args()

    # Constants for training
    NUM_EPOCHS = args.epochs
    TRAIN_BATCH_SIZE = args.batch_size
    LEARNING_RATE = 5e-4
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"

    torch.manual_seed(1337)    

    # Pocessing rgb or greyscale images
    rgb = args.rgb

    # Magnitude of the perturbation to be applied to the hash vaules
    # during training [0, 1)
    perturbation = args.perturbation

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
    train_dataset = Hash2ImgDataset(image_paths=os.path.sep.join(['.', '_data', 'train', 'images']), 
                                hash_paths=os.path.sep.join(['.', '_data', 'train', 'hashes.pkl']), 
                                transforms=transform, 
                                hash_func=args.hash_func, 
                                perturbation=perturbation)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)

    # Initialize the model and send it to the proper device
    model = Hash2ImageModel(rgb=rgb, hash_func=args.hash_func)
    model.to(DEVICE)
    
    # Define the path to save the model
    now = datetime.now()
    dt = now.strftime('%Y-%m-%d_%H:%M:%S%')
    if perturbation == 0:
        model_path = os.path.sep.join(['saved_models', f'{args.hash_func}_{args.dataset}_model.pth'])
    else:
        model_path = os.path.sep.join(['saved_models', f'{perturbation}_{args.hash_func}_{args.dataset}_perturbed_model.pth'])
    
    # Training stuff
    num_batches = len(train_loader)
    loss_tracker = []
    min_loss = float('inf')

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_batches)
    model.train()
    print('[INFO] Starting training...')
    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        pbar = tqdm(range(NUM_EPOCHS))
        for i, (hash, image) in pbar:
            # Forward pass and loss calculation
            hash, images = hash.to(DEVICE), image.to(DEVICE)
            pred_imgs = model(hash)
            loss = criterion(images, pred_imgs)
            train_loss += loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f'Batch: {i+1}/{num_batches} Loss: {loss:.4f}')

        train_loss = train_loss.item() / (num_batches)
        loss_tracker.append(train_loss)
        # Save best model
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
        
    # Display loss history
    plt.plot(loss_tracker)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss History')
    plt.show()
