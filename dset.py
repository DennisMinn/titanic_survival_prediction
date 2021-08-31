import pandas as pd
import os

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

"""
Todo: Add isTest param, if false (None, row) else (review.loc[i, 'survived'],
row)


1) drop and seperate survived column
2) convert Pclass to {0,1,2}
3) convert Name to {Mr: 0, Miss: 1, Mrs: 2, Master: 3, other: 4}
4) convert Sex to {male: 0, female: 0}
5) drop Ticket and Cabin
6) embarked.fillna(embarked.mode())
7) age.fillna(age.median())
8) fare.fillna(fare.mode())
9) convert everything to tensors
"""
class titanic_dataset(Dataset):
    def __init__(self, type):
        self.data = load_data('input')[type]
        self.data.drop(columns = ['Ticket', 'Cabin'], inplace = True)

        self.data.Embarked.fillna(self.data.Embarked.mode()[0], inplace = True)
        self.data.Age.fillna(self.data.Age.median(), inplace = True)
        self.data.Fare.fillna(self.data.Fare.median(), inplace = True)

        self.data.Pclass = self.data.Pclass - 1
        self.data.Name = self.data.apply(convert_name, axis = 1)
        self.data.Sex = self.data.apply(convert_sex, axis = 1)
        self.data.Embarked = self.data.apply(convert_embarked, axis = 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.iloc[idx]

def main():
    data = titanic_dataset('train')
    print(data[5])

if __name__ == '__main__':
    main()
