import torch
import torch.nn as nn
from torch.autograd import Function

class DANN(nn.Module):
    def __init__(self) -> None:
        """Constructor for the DANN model."""
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(310, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
        )
        self.label_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 3),
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
        )

    def forward(self, 
                X: torch.Tensor,
                lambda_: float) -> torch.Tensor:
        """Forward pass of the DANN model.

        Args:
            X (torch.Tensor): input data.
            lambda_ (float): the weight of the label loss.
        """
        feature = self.feature_extractor(X)
        label_pred = self.label_predictor(feature)
        feature_rev = ReverseLayer.apply(feature, lambda_)
        domain_pred = self.domain_discriminator(feature_rev)
        return label_pred, domain_pred

class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None