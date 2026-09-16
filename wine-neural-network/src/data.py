import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Standard Scaler Instance
scaler = StandardScaler()

def download_dataset():
    """Downloads Dataset With Sklearn
            Args:
                None:
                return(X, y):
    """
    # Download dataset
    data = load_wine(as_frame=True)
    df = data.frame

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y

def dataset_splitter(X, y):
    """Splits to Dataset
            Args:
                X: Features
                y: Label
                return(X_train, X_test, y_train, y_test):
    """
    # Splits dataset to test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test, y_train, y_test):
    """Normalize to Data
            Args:
                X_train:
                X_test:
                y_train:
                y_test:
                return(X_train_t, X_test_t, y_train_t, y_test_t):
    """
    # Fit and transform x train data
    X_train = scaler.fit_transform(X_train)
    # Just transform x test data fit trained scaler
    X_test  = scaler.transform(X_test)

    # normalize target data with numpy
    y_train = y_train.to_numpy()
    y_test  = y_test.to_numpy()

    # transform to torch tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.int64).to(device)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.int64).to(device)

    return X_train_t, X_test_t, y_train_t, y_test_t

def create_data_loader(X_train_t, X_test_t, y_train_t, y_test_t):
    """Created Data Loader
            Args:
                X_train_t:
                X_test_t:
                y_train_t:
                y_test_t:
                return(train_loader, test_loader):
    """
    train_data = TensorDataset(X_train_t, y_train_t)
    test_data  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False
    )

    return train_loader, test_loader

def prepare_dataset():
    """Prepares Dataset For Training
            Args:
                None:
                return(train_loader, test_loader):
    """
    # Download dataset
    X, y = download_dataset()
    # Split dataset
    X_train, X_test, y_train, y_test = dataset_splitter(X, y)
    # Normalize Data
    X_train_t, X_test_t, y_train_t, y_test_t = normalize_data(X_train, X_test, y_train, y_test)
    # Create Data Loader
    train_loader, test_loader = create_data_loader(X_train_t, X_test_t, y_train_t, y_test_t)
    return train_loader, test_loader
