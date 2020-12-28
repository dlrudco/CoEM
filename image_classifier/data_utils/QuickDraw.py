import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import os

__all__ = ['QuickDraw']


class QuickDraw(Dataset):
    def __init__(self, root='../Data/quickdraw', train=True, transform=None, using_classes=None):
        self.root = os.path.expanduser(root)
        self.train = train
        
        # get class information
        self.classes = ['alaram clock', 'clock', 'bee', 'bird', 'owl', 'cell phone', 'church', 
                        'cow', 'duck', 'dog', 'frog', 'horse', 'keyboard', 'pencil', 'guitar', 
                        'piano', 'train', 'violin', 'clarinet', 'pig']
        
        self.class_to_idx, self.idx_to_class = self._class_setter(self.classes)
        
        # get data from root
        self.images, self.labels = self._images_labels(self.classes)
        
        # process data
        self.train = train
        self.transform = transform
        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label 

    
    def _images_labels(self, classes):
        images, labels = [], []
        
        for idx, name in enumerate(self.classes):
            image_npy = np.load(os.path.join(self.root, name+'.npy'))
            
            for i in range(100000): # only use 100k samples per each class
                labels.append(idx)
            images.append(torch.Tensor(image_npy[:100000]))
        
        images = torch.cat(images) / 255.0
        images = images.view(-1, 1, 28, 28)
        labels = torch.LongTensor(labels)

        # train & test split
        train_indices = torch.LongTensor([i for i in range(100000*20) if i%200!=0])
        test_indices = torch.LongTensor([i for i in range(100000*20) if i%200==0])
        
        if self.train:
            images, labels = images[train_indices], labels[train_indices]
        
        else:
            images, labels = images[test_indices], labels[test_indices]
        
        return images, labels
        
        
    def _class_setter(self, classes):
        class_to_idx, idx_to_class = dict(), dict()
        for i, name in enumerate(classes):
            class_to_idx[name] = i
            idx_to_class[int(i)] = name
            
        return class_to_idx, idx_to_class
