import pytorch_lightning as pl
from src.models.model import UNet2DModelPL
from torch.utils.data import Dataset, DataLoader
from src.data.dataloader import ButterflyDataloader

import torch

@hydra.main(config_path="conf", config_name="config.yaml")
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


    path = "C:/Users/elleh/OneDrive/MachineLearningOperation/project/ML_Ops_stable_diffusion/data/processed/train.pt"

    model = UNet2DModelPL(sample_size,learning_rate)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=log_frequency)
    dataloaders = {'train': DataLoader(dataset=ButterflyDataloader(path=path), batch_size=batch_size, num_workers=workers)}
    trainer.fit(model, dataloaders['train'])

if __name__=="__main__":
    main()
