import torch
import hydra
import logging
import os
import wandb
import numpy as np
from hydra.utils import get_original_cwd

from omegaconf import OmegaConf

from src.data.dataloader import ButterflyDataloader
from src.models.model import UNet2DModelPL
from PIL import Image
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

log = logging.getLogger(__name__)
wandb.init(project='MNIST_Cookie_test')

# python src/models/predict_model.py hydra.job.chdir=True

@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def evaluate(config):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    log.info(f"configuration: \n{OmegaConf.to_yaml(config)}")
    model_checkpoint =  get_original_cwd()+"/models/checkpoint.pth"
    model = UNet2DModelPL()
    state_dict = torch.load(model_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    epoch = 0
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

if __name__ == "__main__":
    evaluate()
