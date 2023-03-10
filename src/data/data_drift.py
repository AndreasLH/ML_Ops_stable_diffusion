import os
import random

import click
import numpy as np
import pandas as pd
import torch
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from google.cloud import storage
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src import _PROJECT_ROOT
from src.data.dataset import ButterflyDataset


def get_reference_data(model, processor, data_path, n_subsample, batch_size):
    dataset = ButterflyDataset(path=data_path)
    assert (
        len(dataset) >= n_subsample
    ), f"n_subsample must not be bigger than {len(dataset)}"
    idxs = random.sample(list(np.arange(len(dataset))), n_subsample)
    dataloader = DataLoader(
        Subset(dataset, idxs), batch_size=batch_size, shuffle=True, num_workers=0
    )
    reference = []
    for batch in tqdm(dataloader):
        imgs = [img for img in batch["images"]]
        inputs = processor(text=None, images=imgs, return_tensors="pt", padding=True)
        img_features = model.get_image_features(inputs["pixel_values"])
        reference.append(img_features)
    reference = torch.concatenate(reference, dim=0)
    reference = pd.DataFrame(reference.detach().numpy())
    return reference


class GeneratedButterflyDataset(Dataset):
    def __init__(self, root_path):
        self.root = root_path
        self.paths = self._get_paths(self.root)
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
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
        return {"images": img}


class GCSDataset(Dataset):
    def __init__(self, bucket_name, folder_name):
        self.bucket_name = bucket_name
        self.folder_name = folder_name
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.bucket_name)
        self.blobs_iterator = self.bucket.list_blobs(prefix=self.folder_name)
        self.blobs = [blob for blob in self.blobs_iterator]
        self.blobs = [blob for blob in self.blobs if ".png" in blob.path]
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, idx):
        cache_path = os.path.join(_PROJECT_ROOT, "cache.png")

        blob = self.blobs[idx]
        blob.download_to_filename(cache_path)
        img = Image.open(os.path.join(cache_path))
        # os.remove(cache_path)
        img = self.img_transforms(img)
        return {"images": img}


def get_current_data(model, processor, data_path, batch_size):
    dataset = GCSDataset("butterfly_jar", "current_data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    current = []
    for batch in tqdm(dataloader):
        imgs = [img for img in batch["images"]]
        inputs = processor(text=None, images=imgs, return_tensors="pt", padding=True)
        img_features = model.get_image_features(inputs["pixel_values"])
        current.append(img_features)
    current = torch.concatenate(current, dim=0)
    current = pd.DataFrame(current.detach().numpy())
    return current


@click.command()
@click.option("--reference_data_path", default="")
@click.option("--current_data_path", type=str)
@click.option("--n_subsample", default=20)
@click.option("--batch_size", default=16)
def main(reference_data_path, current_data_path, n_subsample, batch_size):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    reference_data = get_reference_data(
        model, processor, reference_data_path, n_subsample, batch_size
    )
    current_data = get_current_data(model, processor, current_data_path, batch_size)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(os.path.join(_PROJECT_ROOT, "reference.html"))

    if os.path.exists(os.path.join(_PROJECT_ROOT, "cache.png")):
        os.remove(os.path.join(_PROJECT_ROOT, "cache.png"))


if __name__ == "__main__":
    np.seterr(divide="ignore")
    main()
