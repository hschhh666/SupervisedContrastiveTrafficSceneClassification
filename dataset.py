import numpy as np
import torch
import torchvision.datasets as datasets
import cv2
import os
from tqdm import tqdm


class myImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, memory = False):
        super(myImageFolder,self).__init__(root, transform)
        self.memory = memory
        self.img_in_memory = []
        self.transform = transform
        if self.memory:
            pbar = tqdm(self.imgs,desc='Loading img to memory')
            for path, target in pbar:
                self.img_in_memory.append(self.loader(path))
            print('All images are loaded into memory')
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        if self.memory:
            img = self.img_in_memory[index]
        else:
            img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index