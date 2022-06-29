import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DANN_trainer(object):
    lambda_ = 1
    gamma = 1
    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 data: list,
                 epochs: int, 
                 batch_size: int,
                 lr: float,) -> None:
        """The constructor for the DANN trainer."""
        self.device = device
        self.model = model.to(device)
        self.data = data
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_label = nn.CrossEntropyLoss().to(device)
        self.loss_domain = nn.CrossEntropyLoss().to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self) -> None:
        """The main function for the DANN trainer."""
        for train_set, val_set in self.data:
            self.train(train_set, val_set)

    def train(self, 
              train: Dataset, 
              val: Dataset) -> None:
        """The train step
        
        Args:
            train (Dataset): the train set
            val (Dataset): the val set
        
        Return:
            None
        """
        trainloader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        valloader = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.epochs+1):
            self.model.train()
            for i, (X, y) in enumerate(trainloader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                # forward on train set
                label_pred, domain_pred_t = self.model(X, self.lambda_)
                loss_label = self.loss_label(label_pred, y)
                y_d_t = torch.ones_like(domain_pred_t)
                loss_domain_t = self.loss_domain(domain_pred_t, y_d_t)
                
                # forward on test set
                X_, _ = next(iter(valloader))
                X_ = X_.to(self.device)
                _, domain_pred_v = self.model(X_, self.lambda_)
                y_d_v = torch.zeros_like(domain_pred_v)
                loss_domain_v = self.loss_domain(domain_pred_v, y_d_v)
                
                # final loss
                loss = loss_label + self.gamma * (loss_domain_t + loss_domain_v)
                loss.backward()
                self.optimizer.step()
                
                # print('epoch: {} batch: {} loss: {}'.format(epoch, i, loss.item()))
        
            if epoch % 1 == 0 or epoch == self.epochs:
                self.test(val)


    def test(self, val: Dataset) -> None:
        """the validation step

        Args:
            val (Dataset): the val set
        """
        valloader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        corrects = 0
        for X, y in valloader:
            X, y = X.to(self.device), y.to(self.device)
            label_pred, _ = self.model(X, self.lambda_)
            corrects += (label_pred.max(1)[1] == y).sum().item()
        acc = corrects / len(val)
        print('\ntest acc is {}'.format(acc))