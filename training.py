from dset import TitanicDataset
from model import TitanicModel

import torch
from torch import nn
from torch import optim

def split_dataset(dataset, p):
    split_idx = int(len(dataset) * p)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    return (train_data, val_data)

def do_training(dataset, model, loss_fn, optimizer, n_epochs):
    for epoch in range(n_epochs):
        training_loss = 0.0
        for survi_t, data_t in zip(dataset[0], dataset[1]):
            pred_survi_t = model(data_t)
            loss = loss_fn(pred_survi_t, survi_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss
            # log_metric

def log_metrics():
    pass

def main():
    titanic_dataset = TitanicDataset(is_train = True, batch_size = 64)
    train_data, val_data = split_dataset(dataset = titanic_dataset, p = 0.8)

    model = TitanicModel(titanic_dataset.n_cols)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)

    do_training(train_data, model, loss_fn, optimizer, 1000)

if __name__ == '__main__':
    main()
