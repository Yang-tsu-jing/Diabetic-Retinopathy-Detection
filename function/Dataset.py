import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class trainDataset(Dataset):
    def __init__(self, data_dir, label_dir, transforms = None):
        self.data_dir = data_dir
        self.data = sorted(os.listdir(data_dir))
        self.label = pd.read_csv(label_dir, index_col = 'image')
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        img = Image.open(f'{self.data_dir}/{img_name}').convert('RGB')

        name, _ = img_name.split('.')
        temp1, temp2, _ = name.split('_')
        name = f'{temp1}_{temp2}'
        #print(name)
        label = self.label.at[name, 'level']
        
        if self.transforms:
            img = self.transforms(img)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return img, label_tensor

class testDataset(Dataset):
    def __init__(self, data_dir, transforms = None):
        self.data_dir = data_dir
        self.data = sorted(os.listdir(data_dir))
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        img = Image.open(f'{self.data_dir}/{img_name}').convert('RGB')

        if self.transforms:
            img = self.transforms(img)
        
        return img