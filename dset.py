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
    def __init__(self, isTrain):
        self.isTrain = isTrain
        self.data = load_data('input')['train'] if isTrain else load_data('input')['test']

        self.data.drop(columns = ['Ticket', 'Cabin'], inplace = True)

        self.data.Embarked.fillna(self.data.Embarked.mode()[0], inplace = True)
        self.data.Age.fillna(self.data.Age.median(), inplace = True)
        self.data.Fare.fillna(self.data.Fare.median(), inplace = True)

        self.data.Pclass = self.data.Pclass - 1
        self.data.Name = self.data.apply(convert_name, axis = 1)
        self.data.Sex = self.data.apply(convert_sex, axis = 1)
        self.data.Embarked = self.data.apply(convert_embarked, axis = 1)

        if isTrain:
            self.survived = self.data.Survived
            self.survived = torch.from_numpy(self.survived.values.astype('long'))

            self.data.drop(columns = 'Survived')

        self.data = torch.from_numpy(self.data.values.astype('float32'))

    def __len__(self):
        return int(self.data.shape[0] / 64) + 1

    def __getitem__(self, idx):
        startidx = min(idx*64, self.data.shape[0]-1)
        endidx = min((idx+1)*64, self.data.shape[0]-1)

        if self.isTrain:
            return (self.survived[startidx: endidx + 1], self.data[startidx: endidx + 1])
        return self.data[startidx: endidx + 1]

def main():
    data = TitanicDataset(isTrain = False)
    print(data[5])

if __name__ == '__main__':
    main()
