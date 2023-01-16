import os

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ButterflyDataset
from tests import _PATH_DATA


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}/processed"), reason="Data files not found"
)  # skip this test if dir data/processed does not exist
def test_data_length():
    dataset = ButterflyDataset(os.path.join(_PATH_DATA, "processed/train.pt"))
    assert (
        len(dataset) == 1000
    ), "Dataset did not have the expected number of samples (1000)"  # assert length of dataset is 1000


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}/processed"), reason="Data files not found"
)  # skip this test if dir data/processed does not exist
def test_data_shape():
    dataset = ButterflyDataset(os.path.join(_PATH_DATA, "processed/train.pt"))
    assert torch.all(
        torch.tensor([i["images"].shape == (3, 128, 128) for i in dataset])
    ), "Samples in the dataset does not have the correct shape (3, 128, 128)"
    # assert all images have shape (3, 128, 128) (channel, width, height)


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}/processed"), reason="Data files not found"
)  # skip this test if dir data/processed does not exist
def test_dataloader():
    dataset = ButterflyDataset(os.path.join(_PATH_DATA, "processed/train.pt"))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch = next(iter(dataloader))["images"]
    assert batch.shape == (
        64,
        3,
        128,
        128,
    ), "Batch does not have the expected shape (64, 3, 128, 128)"
    assert (
        batch.dtype == torch.float32
    ), "Batch does not have expected dtype, torch.float32"
