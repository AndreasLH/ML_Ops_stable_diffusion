import torch
from torch.utils.data import Dataset

class ButterflyDataloader(Dataset):
    def __init__(self, path):

        self.images = torch.load(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]

        return {'images': images}
