from tests import _PATH_DATA
import pytest
import os
import torch

from src.data.dataloader import ButterflyDataloader

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed'), reason="Data files not found") # skip this test if dir data/processed does not exist
def test_data_length():
    dataset = ButterflyDataloader(os.path.join(_PATH_DATA, 'processed/train.pt'))
    assert len(dataset) == 1000, "Dataset did not have the expected number of samples (1000)" # assert length of dataset is 1000

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed'), reason="Data files not found") # skip this test if dir data/processed does not exist
def test_data_shape():
    dataset = ButterflyDataloader(os.path.join(_PATH_DATA, 'processed/train.pt'))
    assert torch.all(torch.tensor([i['images'].shape == (3, 128, 128) for i in dataset])) == True, "Samples in the dataset does not have the correct shape (3, 128, 128)" # assert all images have shape (3, 128, 128) (channel, width, height)