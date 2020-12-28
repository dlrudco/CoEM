import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .QuickDraw import *

__all__ = ['quickdraw_setter']

    
def quickdraw_setter(root='../Data/quickdraw', batch_size=128, num_workers=4):
    train_transform_list = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation((-30, 30)),
        transforms.Normalize(mean=0.1767, std=0.3345)
    ]

    test_transform_list = [
        transforms.Normalize(mean=0.1767, std=0.3345)    
    ]
    
    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose(test_transform_list)
    
    # Datasets
    train_set = QuickDraw(root, train=True, transform=train_transforms) # train transform applied
    test_set = QuickDraw(root, train=True, transform=test_transforms) # test transform applied

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloaders = {'train' : train_loader, 'test' : test_loader,}
    dataset_sizes = {'train': train_set.__len__(), 'test' : test_set.__len__()}
    
    return dataloaders, dataset_sizes
