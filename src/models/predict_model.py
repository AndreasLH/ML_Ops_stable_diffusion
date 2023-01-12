import logging
import os

import hydra
import numpy as np
import torch
from diffusers import DDPMPipeline, DDPMScheduler
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from PIL import Image

import wandb
from src.data.dataloader import ButterflyDataloader
from src.models.model import UNet2DModelPL

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

log = logging.getLogger(__name__)
# wandb.init(project='Butterflies')

# python src/models/predict_model.py hydra.job.chdir=True

@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def evaluate(config):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    log.info(f"configuration: \n{OmegaConf.to_yaml(config)}")
    config = config.experiment
    model_checkpoint =  get_original_cwd()+"/models/model.ckpt"
    model = UNet2DModelPL()
    # state_dict = torch.load(model_checkpoint, map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
# C:\Users\andre\OneDriveDTU\MLops 02476\ML_Ops_stable_diffusion\models\lightning_logs\version_0\checkpoints\epoch=2-step=30.ckpt
    model = UNet2DModelPL.load_from_checkpoint(model_checkpoint)
    model.eval()

    images = model.sample(config.hparams.eval_batch_size, config.hparams.seed)

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.hparams.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/hej.png")


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

if __name__ == "__main__":
    evaluate()
