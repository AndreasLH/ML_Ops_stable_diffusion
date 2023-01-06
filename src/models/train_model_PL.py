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
dataloaders = {'train': DataLoader(dataset=ButterflyDataloader(path=path), batch_size=1, num_workers=0)}
trainer.fit(model, dataloaders['train'])
