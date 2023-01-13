import os

import click
import numpy as np
import yaml
from PIL import Image

from src import _PROJECT_ROOT
from src.models.model import UNet2DModelPL


@click.command()
@click.option("--model_dir", default="", help="model's directory")
def evaluate(model_dir):
    with open(
        os.path.join(_PROJECT_ROOT, model_dir, ".hydra", "config.yaml"), "r"
    ) as f:
        conf = yaml.safe_load(f)
    hpms = conf["experiment"]["hyperparameters"]

    for checkpoint in os.listdir(os.path.join(_PROJECT_ROOT, model_dir, "checkpoints")):
        n = hpms["eval_batch_size"]
        root = int(np.sqrt(n))
        assert root**2 == n, "eval_batch_size must be quadratic"
        rows, cols = root, root

        model_path = os.path.join(_PROJECT_ROOT, model_dir, "checkpoints", checkpoint)
        model = UNet2DModelPL.load_from_checkpoint(
            model_path, sample_size=hpms["image_size"]
        )
        images = model.sample(n)
        image_grid = make_grid(images, rows, cols)
        test_dir = os.path.join(_PROJECT_ROOT, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{os.path.basename(model_path)[:-5]}.png")


def make_grid(images, rows, cols):
    w, h = images[0].size

    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    evaluate()
