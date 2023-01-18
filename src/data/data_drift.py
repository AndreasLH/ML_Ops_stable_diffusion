import os
import random

import evidently
import numpy as np
import pandas as pd
import torch
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src import _PROJECT_ROOT
from src.data.dataset import ButterflyDataset

n_subsample = 2
reference_data_path = "data/processed/train.pt"
batch_size = 2
generated_data_path = 'C:/Users/elleh/OneDrive/MachineLearningOperation/billeder'


def get_reference_data(model, processor, data_path, n_subsample, batch_size):
    dataset = ButterflyDataset(path=data_path)
    idxs = random.sample(list(np.arange(len(dataset))), n_subsample)
    dataloader = DataLoader(Subset(dataset, idxs), batch_size=batch_size, shuffle=True, num_workers=0)
    reference = []
    for batch in tqdm(dataloader):
        imgs = [img for img in batch['images']]
        inputs = processor(text=None, images=imgs, return_tensors="pt", padding=True)
        img_features = model.get_image_features(inputs['pixel_values'])
        reference.append(img_features)
    reference = torch.concatenate(reference, dim=0)
    reference = pd.DataFrame(reference.detach().numpy())
    return reference

class GeneratedButterflyDataset(Dataset):
    def __init__(self, root_path):
        self.root = os.path.join(_PROJECT_ROOT, root_path)
        self.paths = self._get_paths(self.root)
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((128, 128)),  # config.image_size = 128
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalise pixel values to the range [-1,1]
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _get_paths(self, root):
        paths = []
        for path in os.listdir(root):
            paths.append(os.path.join(root, path))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil_img = Image.open(path)
        img = self.img_transforms(pil_img)
        return {'images': img}

def get_current_data(model, processor, data_path, batch_size):
    dataset = GeneratedButterflyDataset(root_path=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    current = []
    for batch in dataloader:
        imgs = [img for img in batch['images']]
        inputs = processor(text=None, images=imgs, return_tensors="pt", padding=True)
        img_features = model.get_image_features(inputs['pixel_values'])
        current.append(img_features)
    current = torch.concatenate(current, dim=0)
    current = pd.DataFrame(current.detach().numpy())
    return current






if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    reference_data = get_reference_data(model, processor, reference_data_path, n_subsample, batch_size)
    current_data = get_current_data(model, processor, generated_data_path, batch_size)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(os.path.join(_PROJECT_ROOT, 'reference.html'))







