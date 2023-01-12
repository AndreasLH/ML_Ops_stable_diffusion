import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

import wandb
from src.data.dataset import ButterflyDataset
from src.models.model import UNet2DModelPL

wandb_logger = WandbLogger(name="Oldehammer-Master", project="mlopsproject21")
path = os.path.join(os.getcwd(), "data/processed/train.pt")

model = UNet2DModelPL()
trainer = pl.Trainer(max_epochs=2, log_every_n_steps=2, logger=wandb_logger)

import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ButterflyDataset
from src.models.model import UNet2DModelPL


@hydra.main(config_path="../../conf/", config_name="config.yaml")
def main(cfg):
    hpms = cfg.experiment['hyperparameters']
    seed = hpms.seed
    epochs = hpms.num_epochs
    log_frequency = hpms.log_frequency
    learning_rate = hpms.learning_rate
    sample_size = hpms.sample_size
    batch_size = hpms.train_batch_size
    workers = hpms.workers

    torch.manual_seed(seed) #Set seed

path = os.path.join(os.getcwd(),"data/processed/train.pt")

model = UNet2DModelPL()
trainer = pl.Trainer(max_epochs=3, log_every_n_steps=2, default_root_dir="models/path/")

dataloaders = {'train': DataLoader(dataset=ButterflyDataset(path=path), batch_size=1, num_workers=0)}
trainer.fit(model, dataloaders['train'])

    model = UNet2DModelPL(sample_size,learning_rate)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=log_frequency, default_root_dir="models/path/")
    dataloaders = {'train': DataLoader(dataset=ButterflyDataset(path=path), batch_size=batch_size, num_workers=workers)}
    trainer.fit(model, dataloaders['train'])

if __name__=="__main__":
    main()
