from dset import TitanicDataset
from model import TitanicModel

import torch
from torch import nn
from torch import optim

class TitanicApp:
    def __init__(self):
        self.model = None
        self.loss_fn = None
        self.optimizer = None

    def init_model(self, in_chans = 8):
        self.model = TitanicModel(in_chans)

    def init_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def init_optimizer(self, lr = 1e-2):
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

    def do_training(self, n_epochs, train_dl):
        for epoch in range(n_epochs):
            training_loss = 0.0
            for survi_t, data_t in zip(dataset[0], dataset[1]):
                pred_survi_t = self.model(data_t)
                loss = self.loss_fn(pred_survi_t, survi_t)

                self.optimizer.zero_grad()
                self.loss_fn.backward()
                self.optimizer.step()

                training_loss += loss
                # log_metrics

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

    def main(self):
        titanic_dataset = TitanicDataset(is_train = True, batch_size = 64)
	train_data, val_data = split_dataset(dataset = titanic_dataset, p = 0.8)

        init_model()
        init_optimizer()
        init_loss()

        #do_training
