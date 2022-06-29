import torch.nn as nn

# import internal libs
from utils import get_logger


def prepare_model(model: str,
                  dataset: str,) -> nn.Module:
    """prepare the model
    
    Args:
        model (str): the name of the model
        dataset (str): the name of the dataset
    
    Return:
        the model
    """
    logger = get_logger(__name__)
    logger.info(f"prepare the model {model}")
    if model == "DANN" and dataset == "SEED":
        from model.DANN import DANN
        model = DANN()
    else:
        raise NotImplementedError(f"the model {model} is not implemented.")
    return model