import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split

class DiverseDermatologyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 4]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataset_splits(csv_file, root_dir, transform, split_ratio=0.8):
    dataset = DiverseDermatologyDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset