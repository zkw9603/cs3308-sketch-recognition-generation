# src/recognition/dataset.py

import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

class Binarize(object):
    def __call__(self, img):
        return (img > 0.1).float()

def get_dataloaders(data_dir, batch_size=32, image_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        Binarize(),                          
        transforms.Lambda(lambda x: 1.0 - x), 
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])
    
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset.classes