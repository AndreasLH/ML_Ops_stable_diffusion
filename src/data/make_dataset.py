# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import torch
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath : str) -> None:
    """
    Runs data processing scripts to turn the raw Smithsonian butterfly dataset from huggan into
        cleaned data ready to be analyzed (saved in ../processed).
    :param output_filepath: path where the processed data is outputted
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(dataset_name, split="train")

    # Feel free to try other datasets from https://hf.co/huggan/ too!
    # Here's is a dataset of flower photos:
    # config.dataset_name = "huggan/flowers-102-categories"
    # dataset = load_dataset(config.dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # config.image_size = 128
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalise pixel values to the range [-1,1]
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    train_ = [dataset[i]['images'] for i in range(len(dataset))]
    train = torch.stack(train_)

    torch.save(train, os.path.join(output_filepath, 'train.pt'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
