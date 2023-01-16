import os

import click
import numpy as np
import yaml
from PIL import Image
import torch

from src import _PROJECT_ROOT
from src.models.model import UNet2DModelPL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option("--model_dir", default="", help="model's directory")
def evaluate(model_dir):
    eval(model_dir)


def eval(model_dir, steps=None, n_images=None):
    with open(
        os.path.join(_PROJECT_ROOT, model_dir, ".hydra", "config.yaml"), "r"
    ) as f:
        conf = yaml.safe_load(f)
    hpms = conf["experiment"]["hyperparameters"]
    if steps is not None:
        hpms["num_inference_steps"] = steps
    if n_images is not None:
        hpms["eval_batch_size"] = n_images

    for checkpoint in os.listdir(os.path.join(_PROJECT_ROOT, model_dir, "checkpoints")):
        n = hpms["eval_batch_size"]
        root = int(np.sqrt(n))
        assert root**2 == n, "eval_batch_size must be quadratic"
        rows, cols = root, root

        model_path = os.path.join(_PROJECT_ROOT, model_dir, "checkpoints", checkpoint)
        model = UNet2DModelPL.load_from_checkpoint(
            model_path, sample_size=hpms["image_size"]
        )
        images = model.sample(
            batch_size=n,
            seed=hpms["seed"],
            num_inference_steps=hpms["num_inference_steps"],
        )
        image_grid = make_grid(images, rows, cols)
        test_dir = os.path.join(_PROJECT_ROOT, "samples")
        os.makedirs(test_dir, exist_ok=True)
        save_point = f"{test_dir}/{os.path.basename(model_path)[:-5]}.png"
        image_grid.save(save_point)
        return save_point

@click.command()
@click.option("--model_name", default="", help="model's directory")
@click.option("--steps", default=3, help="model's directory")
def eval2(model_name, steps=None, n_images=None, seed=0):
    with open(
        os.path.join(_PROJECT_ROOT, "conf", "experiment", "train_conf.yaml"), "r"
    ) as f:
        conf = yaml.safe_load(f)
    hpms = conf["hyperparameters"]
    if steps is not None:
        hpms["num_inference_steps"] = steps
    if n_images is not None:
        hpms["eval_batch_size"] = n_images

    n = hpms["eval_batch_size"]
    root = int(np.sqrt(n))
    assert root ** 2 == n, "eval_batch_size must be quadratic"
    rows, cols = root, root

    model_path = os.path.join(_PROJECT_ROOT, "models", model_name)
    model = UNet2DModelPL.load_from_checkpoint(
        model_path, sample_size=hpms["image_size"]
    )
    images = model.sample(
        batch_size=n,
        seed=hpms["seed"],
        num_inference_steps=hpms["num_inference_steps"],
    )
    image_grid = make_grid(images, rows, cols)
    # image_grid.show()
    test_dir = os.path.join(_PROJECT_ROOT, "samples")
    os.makedirs(test_dir, exist_ok=True)
    save_point = f"{test_dir}/{os.path.basename(model_path)[:-5]}.png"
    image_grid.save(save_point)




def eval_gcs(model_dir, steps=None, n_images=None, seed=0):

    n = n_images
    root = int(np.sqrt(n))
    assert root**2 == n, "eval_batch_size must be quadratic"
    rows, cols = root, root

    model_path = model_dir
    model = UNet2DModelPL.load_from_checkpoint(
        model_path, sample_size=128
    )
    model = model.to(device)
    images = model.sample(
        batch_size=n,
        seed=seed,
        num_inference_steps=steps,
    )
    image_grid = make_grid(images, rows, cols)
    test_dir = "/gcs/model_best/samples"
    os.makedirs(test_dir, exist_ok=True)
    save_point = f"{test_dir}/{os.path.basename(model_path)[:-5]}.png"
    image_grid.save(save_point)
    return save_point


def make_grid(images, rows, cols):
    w, h = images[0].size

    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    eval2()
    # evaluate()
    # eval2('outputs/2023-01-16/14-11-58')

