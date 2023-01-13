import torch
import pytest

from src.models.train_model_PL import UNet2DModelPL
from src.data.dataset import ButterflyDataset
import os
from tests import _PATH_DATA
from torch.utils.data import DataLoader


@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed'), reason="Data files not found")  # skip this test if dir data/processed does not exist

def test_training_loop_PL():
    dataset = ButterflyDataset(os.path.join(_PATH_DATA, 'processed/train.pt'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    model = UNet2DModelPL(16,1e-3,16)
    loss = model.training_step(batch, batch_idx=1)
    assert loss.dtype == torch.float32