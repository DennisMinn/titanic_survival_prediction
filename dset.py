import os
import pandas as pd

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

def load_data(path):
    data = {}
    data['train'] = pd.read_csv(os.path.join(path, 'train.csv'), index_col = 'PassengerId')
    data['test'] = pd.read_csv(os.path.join(path, 'test.csv'), index_col ='PassengerId')

    return data

def convert_name(row):
    if 'Mr.' in row.Name:
        return 0
    elif 'Miss.' in row.Name:
        return 1
    elif 'Mrs.' in row.Name:
        return 2
    elif 'Master.' in row.Name:
        return 3
    else:
        return 4

def convert_sex(row):
    return row.Sex == 'male'

def convert_embarked(row):
    if 'Q' in row.Embarked:
        return 0
    elif 'S' in row.Embarked:
        return 1
    else:
        return 2

class TitanicDataset(Dataset):
    def __init__(self, is_train = True, batch_size = 64):
        self.is_train = is_train
        self.data = load_data('input')['train'] if is_train else load_data('input')['test']

        self.data.drop(columns = ['Ticket', 'Cabin'], inplace = True)

        self.data.Embarked.fillna(self.data.Embarked.mode()[0], inplace = True)
        self.data.Age.fillna(self.data.Age.median(), inplace = True)
        self.data.Fare.fillna(self.data.Fare.median(), inplace = True)

        self.data.Pclass = self.data.Pclass - 1
        self.data.Name = self.data.apply(convert_name, axis = 1)
        self.data.Sex = self.data.apply(convert_sex, axis = 1)
        self.data.Embarked = self.data.apply(convert_embarked, axis = 1)

        if is_train:
            self.y = self.data.Survived
            self.y = torch.from_numpy(self.y.values).long()
            self.y = self.y.split(64)

            self.data.drop(columns = 'Survived', inplace = True)

        self.n_cols = self.data.shape[1]
        self.X = torch.from_numpy(self.data.values.astype('float32'))
        self.X = self.X.split(batch_size)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_train:
            return self.y[idx], self.X[idx]
        else:
            return self.X[idx]

