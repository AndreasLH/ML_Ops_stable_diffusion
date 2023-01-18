import os

import optuna
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from src import _PROJECT_ROOT
from src.models.model import UNet2DModelPL


def objective(trial):
    params = {
        'learning rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-1),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    }

    cfg = yaml.safe_load(open("../../conf/experiment/train_conf.yaml"))

    hpms = cfg["hyperparameters"]
    seed = hpms['seed']
    epochs = hpms['num_epochs']
    log_frequency = hpms['log_frequency']
    learning_rate = hpms['learning_rate']
    image_size = hpms['image_size']
    batch_size = hpms['train_batch_size']
    validation_n_samples = hpms['validation_n_samples']
    name = hpms['experiment_name']

    workers = hpms['workers']

    torch.manual_seed(seed)  # Set seed

    path = os.path.join(_PROJECT_ROOT, hpms['datapath'])

    model = UNet2DModelPL(image_size, learning_rate, hpms)

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
    )

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
    hyperparameters = dict(lr=params['learning rate'], optimizer=params['optimizer'])


    trainer.fit(model, dataloaders["train"], dataloaders["val"])


    return trainer.callback_metrics['train_loss'].item()

if __name__ == '__main__':
    cfg = yaml.safe_load(open("../../conf/experiment/train_conf.yaml"))

    # create study, where the objective is to minimise and sample with Bayesian optimisation
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))