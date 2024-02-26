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
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', help='Are the images in the dataset rgb or greyscale', default=0, type=int)
    parser.add_argument('--epochs', help='Number of Epochs to train for', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size to be used in training', default=32, type=int)
    parser.add_argument('--learning_rate', help='Learning Rate to be used during training', default=5e-4, type=float)
    args = parser.parse_args()

    # Constants for training
    NUM_EPOCHS = args.epochs
    TRAIN_BATCH_SIZE = args.batch_size
    LEARNING_RATE = 5e-4
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

    torch.manual_seed(1337)

    

    # Pocessing rgb o greyscale images
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
    train_dataset = Hash2ImgDataset(image_paths='./_data/train/images', hash_paths='./_data/train/hashes.pkl', transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)

    # Initialize the model and send it to the proper device
    model = Hash2ImageModel()
    model.to(DEVICE)
    
    # Define the path to save the model
    now = datetime.now()
    dt = now.strftime('%Y-%m-%d_%H:%M:%S%')
    model_path = f'saved_models/{dt}_saved_model.pth'
    
    # Training stuff
    early_stop = False
    num_batches = 100 if early_stop else len(train_loader)
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
        for i, (hash, image) in enumerate(train_loader):
            if i == num_batches:
                break
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
            print(f'Batch: {i+1}/{num_batches} Loss: {loss:.4f}')

        train_loss = train_loss.item() / (num_batches)
        loss_tracker.append(train_loss)
        # Save best model
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        print(f'\nEpoch: {epoch+1}/{NUM_EPOCHS} Avg Loss: {train_loss:.4f}\n')
        
    print('[INFO] Training complete')

    # Display loss history
    plt.plot(loss_tracker)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss History')
    plt.show()
