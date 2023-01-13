import torch
import pytest

from src.models.train_model_PL import UNet2DModelPL
from src.data.dataset import ButterflyDataset
import os
from tests import _PATH_DATA
from torch.utils.data import DataLoader

class testClass():
    def __init__(self):
        self.eval_batch_size = 8
        self.seed = 123
hpms = testClass()
model = UNet2DModelPL(32, 1e-3, hpms)

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed'), reason="Data files not found")  # skip this test if dir data/processed does not exist

def test_training_loop_PL():
    dataset = ButterflyDataset(os.path.join(_PATH_DATA, 'processed/train.pt'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    loss = model.training_step(batch, batch_idx=1)
    assert loss.dtype == torch.float32


#@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed'), reason="Data files not found")  # skip this test if dir data/processed does not exist
#def test_evaluation_PL():
#    dataset = ButterflyDataset(os.path.join(_PATH_DATA, 'processed/train.pt'))
#    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
#    batch = next(iter(dataloader))
#    Inception_score = model.validation_step(batch, batch_idx=1)
#    assert Inception_score.dtype == torch.float32
