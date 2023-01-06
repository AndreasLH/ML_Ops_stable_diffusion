import pytorch_lightning as pl
from src.models.model import UNet2DModelPL
from torch.utils.data import Dataset, DataLoader
from src.data.dataloader import ButterflyDataloader

import torch

class Dummy(Dataset):
    def __init__(self):
        self.data = torch.randn((10, 3, 128, 128))
    def __getitem__(self, idx):
        return {'images': self.data[idx]}
    def __len__(self):
        return self.data.shape[0]

path = "C:/Users/elleh/OneDrive/MachineLearningOperation/project/ML_Ops_stable_diffusion/data/processed/train.pt"

model = UNet2DModelPL()
trainer = pl.Trainer(max_epochs=3, log_every_n_steps=2)
dataloaders = {'train': DataLoader(dataset=ButterflyDataloader(path=path), batch_size=1, num_workers=0)}
trainer.fit(model, dataloaders['train'])
