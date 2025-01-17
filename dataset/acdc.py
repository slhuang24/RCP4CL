from .transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box
from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ACDCDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'val':
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        else:
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        if self.mode == 'val' or self.mode == 'test':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s =  deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = blur(img_s, p=0.5)
        cutmix_box = obtain_cutmix_box(self.size, p=0.5)
        img_s = torch.from_numpy(np.array(img_s)).unsqueeze(0).float() / 255.0

        return img, img_s

    def __len__(self):
        return len(self.ids)
