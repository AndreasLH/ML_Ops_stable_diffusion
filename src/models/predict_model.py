from src.models.model import UNet2DModelPL
from src import _PROJECT_ROOT
import os
import hydra
from PIL import Image


@hydra.main(version_base=None, config_path=os.path.join(_PROJECT_ROOT, 'conf'), config_name="config.yaml")
def evaluate(conf):
    hpms = conf.experiment['hyperparameters']
    path = os.path.join(_PROJECT_ROOT, 'outputs/2023-01-12/14-20-40/mlopsproject21/2fpleg5r/checkpoints/epoch=0-step=1.ckpt')
    model = UNet2DModelPL.load_from_checkpoint(path, sample_size=hpms.image_size)

    images = model.sample(hpms.eval_batch_size)

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(_PROJECT_ROOT, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/output.png")

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

evaluate()
