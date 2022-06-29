import pickle
from re import X
import numpy as np
import torch
from torch.utils.data import Dataset

# import internal libs
from utils import get_logger

class SEEDDataset(Dataset):
    """SEED dataset."""

    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray) -> None:
        """
        Args:
            X (np.ndarray): the input data.
            y (np.ndarray): the target data
        """
        self.X = X
        self.y = y
        if not isinstance(self.X, torch.Tensor):
            self.X = torch.from_numpy(self.X).float()
        if not isinstance(self.y, torch.Tensor):
            self.y = torch.from_numpy(self.y).long()

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]


def load(root: str = "../data/SEED/data.pkl",
         val_domain: int = 0) -> tuple:
    """prepare the dataset SEED

    Args:
        root (str): the root path of the dataset.
        val_domain (int): the validation domain

    Return:
        train set and val set (include input, label)
    """
    logger = get_logger(__name__)

    # load the pickle file
    with open(root, "rb") as f:
        data = pickle.load(f)

    # preprocessing
    for d in data.keys():
        data[d]["label"] += 1

    # get the train set and test set
    val_domain = f"sub_{val_domain}"
    train_X = np.vstack([v['data'] for k, v in data.items() if k != val_domain])
    train_Y = np.hstack([v['label'] for k, v in data.items() if k != val_domain])
    val_X = data[val_domain]["data"]
    val_Y = data[val_domain]["label"]
    
    # basic info
    logger.info(f"train_X shape: {train_X.shape}; train_Y shape: {train_Y.shape}")
    logger.info(f"val_X shape: {val_X.shape}; val_Y shape: {val_Y.shape}")

    # get the train set and test set
    train_set = SEEDDataset(train_X, train_Y)
    val_set = SEEDDataset(val_X, val_Y)
    return train_set, val_set