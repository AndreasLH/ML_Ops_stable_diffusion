import torch
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    def __init__(self, path: str) -> None:
        """

        :param path: path to where the processed dataset is located
        """
        self.images = torch.load(path)

    def __len__(self) -> int:
        """
        Allows len to be used on ButterflyDataset
        :return: length of dataset
        """
        # return len(self.images)
        return 30

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Allows ButterflyDataset to be indexed.
        :param idx: index
        :return: image corresponding to the idx as torch Tensor.
        """
        images = self.images[idx]

        return {"images": images}


class ValidationDataset(Dataset):
    def __init__(self, n_samples: int):
        """
        :param n_samples: number of samples to be generated during validation
        """
        self.n_samples = n_samples

    def __len__(self):
        """
        Allows len to be used on ValidationDataset
        :return: length of dataset
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        :param idx: index
        :return: just returning None as this is just a dummy data set
        """
        return {"images": torch.randn((0, 3, 128, 128))}
