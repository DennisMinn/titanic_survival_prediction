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
    def __init__(self, is_train, batch_size):
        self.is_train = is_train
        self.data = load_data('input')['train'] if is_train else load_data('input')['test']

        self.data.drop(columns = ['Ticket', 'Cabin'], inplace = True)

        self.n_cols = self.data.shape[1]

        self.data.Embarked.fillna(self.data.Embarked.mode()[0], inplace = True)
        self.data.Age.fillna(self.data.Age.median(), inplace = True)
        self.data.Fare.fillna(self.data.Fare.median(), inplace = True)

        self.data.Pclass = self.data.Pclass - 1
        self.data.Name = self.data.apply(convert_name, axis = 1)
        self.data.Sex = self.data.apply(convert_sex, axis = 1)
        self.data.Embarked = self.data.apply(convert_embarked, axis = 1)

        if is_train:
            self.survived = self.data.Survived
            self.survived = torch.from_numpy(self.survived.values.astype('long'))
            self.survived = self.survived.split(64)

            self.data.drop(columns = 'Survived')

        self.data = torch.from_numpy(self.data.values.astype('float32'))
        self.data = self.data.split(64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            return self.survived[idx], self.data[idx]
        else:
            return self.data[idx]

def main():
    data = TitanicDataset(is_train = False)
    print(data[5])

if __name__ == '__main__':
    main()
