import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

_CURRENT_PATH = Path(__file__).parent.parent
_DATA_PATH    = _CURRENT_PATH / "data"
_FILE_PATH    = _DATA_PATH / "train_and_test2.csv"

# Standard Scaler
scaler = StandardScaler()

def read_file():
    """Reads File
    """
    # read csv file
    df   = pd.read_csv(_FILE_PATH)

    # get copy for work on it
    data = df.copy()

    # select usiable columns
    data = df[['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked', '2urvived']]
    data = data.rename(columns={"2urvived": "survived"})

    # find popular value and fix missing value error
    embarked_mode = data['Embarked'].mode()[0]
    data['Embarked'] = data['Embarked'].fillna(embarked_mode)

    X = data.drop(columns=["survived"])
    Y = data["survived"]
    
    return X, Y

def split_test_and_train(X, Y):
    """Splits data for test and train
            Args:
                X(list): The list of the values
                Y(list): The list of the labels
    """
    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # transform x test and x train data
    X_train = scaler.fit_transform(X_train[['Age', 'Fare', 'Pclass', 'Embarked', 'Sex', 'sibsp', 'Parch']])
    X_test = scaler.transform(X_test[['Age', 'Fare', 'Pclass', 'Embarked', 'Sex', 'sibsp', 'Parch']])

    # transform y train and y test data to numpy array because we did not transform with standard scaler
    y_train = y_train.to_numpy()
    y_test  = y_test.to_numpy()

    # cpu to cuda
    x_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)

    x_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

    # create tensor datasets from train and test dataset
    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset  = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32, # I think it is enought
        shuffle=True # because we want to model learn better
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32, # same with train data
        shuffle=False # because it is only test data
    )

    return train_loader, test_loader

