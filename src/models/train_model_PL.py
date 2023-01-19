import os

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src import _PROJECT_ROOT
from src.data.dataset import ButterflyDataset, ValidationDataset
from src.models.model import UNet2DModelPL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base="1.2",
    config_path=os.path.join(_PROJECT_ROOT, "conf"),
    config_name="config.yaml",
)
def main(cfg):
    hpms = cfg.experiment["hyperparameters"]
    seed = hpms.seed
    epochs = hpms.num_epochs
    log_frequency = hpms.log_frequency
    learning_rate = hpms.learning_rate
    image_size = hpms.image_size
    batch_size = hpms.train_batch_size
    validation_n_samples = hpms.validation_n_samples
    name = hpms.experiment_name

    workers = hpms.workers

    torch.manual_seed(seed)  # Set seed

    path = os.path.join(_PROJECT_ROOT, hpms.datapath)

    model = UNet2DModelPL(image_size, learning_rate, hpms)
    model = model.to(device)
    if hpms.wandb_log:
        logger = WandbLogger(name=name, project="mlopsproject21", entity="mlopsproject21")
    else:
        logger = False
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"
    checkpoint_callback = ModelCheckpoint(
        dirpath=hpms.output_dir,
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        every_n_epochs=1,
        filename="best",
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=log_frequency,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        check_val_every_n_epoch=hpms.check_val_every_n_epoch,
        num_sanity_val_steps=0,
    )
    # todo: vi skal have en val dataloader som ikke bare er det samme som train dataloaderen
    dataloaders = {
        "train": DataLoader(
            dataset=ButterflyDataset(path=path),
            batch_size=batch_size,
            num_workers=workers,
            shuffle=True,
        ),
        "val": DataLoader(
            dataset=ValidationDataset(n_samples=validation_n_samples),
            batch_size=batch_size,
            num_workers=workers,
        ),
    }
    trainer.fit(model, dataloaders["train"], dataloaders["val"])


if __name__ == "__main__":
    main()
