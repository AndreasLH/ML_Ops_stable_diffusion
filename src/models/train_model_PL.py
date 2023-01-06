import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule
from src.models.model import UNet2DModelPL
from torch.utils.data import Dataset, DataLoader
from src.data.dataloader import ButterflyDataloader
import wandb
import torch

wandb_logger = WandbLogger(name="Oldehammer-Master", project="mlopsproject21")
path = os.path.join(os.getcwd(), "data/processed/train.pt")

model = UNet2DModelPL()
trainer = pl.Trainer(max_epochs=2, log_every_n_steps=2, logger=wandb_logger)
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

path = "data/processed/train.pt"

# dataset = Dummy()

model = UNet2DModelPL()
trainer = pl.Trainer(max_epochs=3, log_every_n_steps=2, default_root_dir="models/path/")

dataloaders = {'train': DataLoader(dataset=ButterflyDataloader(path=path), batch_size=1, num_workers=0)}
trainer.fit(model, dataloaders['train'])
