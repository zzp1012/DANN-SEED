import torch
import torch.nn as nn

# import internal libs
from model.DANN import DANN

def train(device: torch.device,
          model: nn.Module,
          data: list,
          epochs: int, 
          batch_size: int,
          lr: float,) -> None:
    """train the model

    Args:
        device (torch.device): GPU.
        model (nn.Module): the model
        data (list): training and validation data
        epochs (int): the training epochs
        batch_size (int): batch size
        lr (float): learning rate

    Return:
        None
    """
    
    if isinstance(model, DANN):
        from trainer.DANN import DANN_trainer
        trainer = DANN_trainer(device, model, data, epochs, batch_size, lr)
        trainer.run()
    else:
        raise NotImplementedError