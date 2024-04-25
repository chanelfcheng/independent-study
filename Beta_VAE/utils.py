import torch
from torch.utils.data import Dataset
import numpy as np

class TeapotsDatasetNPZ(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.images = data['images']
        self.gts = data['gts']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1) / 255.0
        gts = self.gts[idx]
        if self.transform:
            image = self.transform(image)
        return image, gts