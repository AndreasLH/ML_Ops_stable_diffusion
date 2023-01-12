import torch
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    def __init__(self, path : str) -> None:
        """

        :param path: path to where the processed dataset is located
        """
        self.images = torch.load(path)

    def __len__(self) -> int:
        """
        Allows len to be used on ButterflyDataset
        :return: length of dataset
        """
        return len(self.images)

    def __getitem__(self, idx : int) -> torch.Tensor:
        """
        Allows ButterflyDataset to be indexed.
        :param idx: index
        :return: image corresponding to the idx as torch Tensor.
        """
        images = self.images[idx]

        return {'images': images}
