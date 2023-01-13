import os

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from src import _PATH_DATA, _PROJECT_ROOT
from torch.utils.data import DataLoader, Dataset

from src import _PATH_DATA, _PROJECT_ROOT
from src.data.dataset import ButterflyDataset
from src.models.model import UNet2DModelPL


@hydra.main(config_path=os.path.join(_PROJECT_ROOT, "conf"), config_name="config.yaml")
def main(cfg):
    hpms = cfg.experiment["hyperparameters"]
    seed = hpms.seed
    epochs = hpms.num_epochs
    log_frequency = hpms.log_frequency
    learning_rate = hpms.learning_rate
    image_size = hpms.image_size
    batch_size = hpms.train_batch_size
    workers = hpms.workers

    torch.manual_seed(seed)  # Set seed

    path = os.path.join(_PATH_DATA, "processed/train.pt")

    model = UNet2DModelPL(image_size, learning_rate)
    logger = WandbLogger(name="Oldehammer-Master", project="mlopsproject21")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        monitor="train_loss",
        every_n_epochs=1,
        filename="{epoch}-{train_loss:.12f}",
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=log_frequency,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    # todo: vi skal have en val dataloader som ikke bare er det samme som train dataloaderen
    dataloaders = {
        "train": DataLoader(
            dataset=ButterflyDataset(path=path),
            batch_size=batch_size,
            num_workers=workers,
        ),
        "val": DataLoader(
            dataset=ButterflyDataset(path=path),
            batch_size=batch_size,
            num_workers=workers,
        ),
    }
    trainer.fit(model, dataloaders["train"], dataloaders["val"])


if __name__ == "__main__":
    main()
