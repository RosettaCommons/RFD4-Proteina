"""Mock DataModule that always returns the same atomized batch for overfitting tests."""

import pickle

from torch.utils.data import Dataset


class MockAtomworksDataset(Dataset):
    """
    Dataset that always returns the same pickled batch.

    Note the batch should be made before collation. Here we just collate a bunch of copies together
    to make a full batch.
    """

    def __init__(self, data: str, num_samples: int = 100, name: str = "mock_atomworks", **kwargs):
        """
        Args:
            data: Path to the pickled batch. Note this is pre-collation!
            num_samples: Number of samples to return
            name: Name of the dataset
            **kwargs: Dumped for now
        """
        self.name = name
        with open(data, "rb") as f:
            self.batch = pickle.load(f)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Return the batch"""
        return self.batch
