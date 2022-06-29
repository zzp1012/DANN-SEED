import pickle
import numpy as np
import torch

# import internal libs
from utils import get_logger

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

    # convert to torch.Tensor
    train_X = torch.from_numpy(train_X).float()
    train_Y = torch.from_numpy(train_Y).long()
    val_X = torch.from_numpy(val_X).float()
    val_Y = torch.from_numpy(val_Y).long()

    # basic info
    logger.info(f"train_X shape: {train_X.shape}; train_Y shape: {train_Y.shape}")
    logger.info(f"val_X shape: {val_X.shape}; val_Y shape: {val_Y.shape}")
    
    return (train_X, train_Y), (val_X, val_Y)