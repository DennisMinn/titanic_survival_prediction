from dset import TitanicDataset
from model import TitanicModel

import torch
from torch import nn
from torch import optim

from sklearn.metrics import a

class TitanicApp:
    def __init__(self):
        self.model = None
        self.loss_fn = None
        self.optimizer = None

    def init_model(self, model):
        self.model = model

    def init_loss(self, loss_fn):
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def init_optimizer(self, optimizer = None, lr = 1e-2):
        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr = lr)
        else:
            self.optimizer = optimizer

    def do_training(self, n_epochs, train_dl):
        for epoch in range(n_epochs):
            training_loss = 0.0
            for survi_t, data_t in train_dl:
                self.optimizer.zero_grad()

                pred_survi_t = self.model(data_t)

                loss = self.loss_fn(pred_survi_t, survi_t)
                loss.backward()
                self.optimizer.step()

                training_loss += loss
            if epoch % 10 == 0:
                print(training_loss)

    def evaluate(self, val_dl):
        pass

    def log_metrics(self):
        pass

    def get_params(self):
        pass

    def split_dataset(self, dataset, p):
        split_idx = int(len(dataset) * p)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]

        return (train_data, val_data)
