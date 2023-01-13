import os

import pytest
import torch
# from models import UNet2DmodelPL
from diffusers import UNet2DModel

from tests import _PROJECT_ROOT

# create inputs to model output size test
test_input = [{"input": torch.ones(1, 3, i, i), "output_size": i} for i in [32, 64, 128, 256]]


@pytest.mark.parametrize("test_input,expected", [(test_input[i], (1, 3, test_input[i]['output_size'], test_input[i]['output_size'])) for i in range(len(test_input))])
def test_model_output(test_input, expected):
    model = UNet2DModel(
        sample_size=test_input["output_size"],  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )
    assert model(test_input["input"], timestep=5).sample.shape == torch.Size(expected)
