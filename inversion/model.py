import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from hash import compute_hash, hash2tensor

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        orig_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + orig_x
        x = self.dropout(x)
        return x
    
class Hash2ImageModel(nn.Module):
    def __init__(self, rgb=True, hash_func='pdq'):
        super().__init__()
        if hash_func == 'neuralhash':
            self.linear = nn.Linear(128, 1024)
        elif hash_func == 'photodna':
            self.linear = nn.Linear(144, 1024)
        else:
            self.linear = nn.Linear(256, 1024)
        self.conv1 = nn.Conv2d(1, 64, 3, padding='same')
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 5, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 3, 5, stride=4) if rgb else nn.Conv2d(64, 1, 5, stride=4)
        
    def forward(self, x):
        x = x.type(torch.float32)
        x = self.linear(x)
        x = x.view(x.size(0), 1, 32, 32)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.conv2(x)
        return x
    
if __name__ == '__main__':
    img_path = os.path.sep.join(['.', '_data', 'train', 'images', '1.jpeg'])
    img = Image.open(img_path)
    hash = compute_hash(np.array(img))
    hash_tensor = hash2tensor(hash)
    model = Hash2ImageModel()
    output = model(hash_tensor)
    print(output.shape)


