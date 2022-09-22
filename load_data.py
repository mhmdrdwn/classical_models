import numpy as np
from tqdm import tqdm
import glob
import random

from utils import StandardScaler3D


def data_file_names():
    test_abnormal_files = glob.glob('./tuh/eval/abnormal/*.npy')
    test_normal_files = glob.glob('./tuh/eval/normal/*.npy')
    normal_files = glob.glob('./tuh/train/normal/*.npy')
    abnormal_files = glob.glob('./tuh/train/abnormal/*.npy')

    sample_size = 400
    train_data_size = 300
    normal_files = random.sample(normal_files, sample_size)
    abnormal_files = random.sample(abnormal_files, sample_size)

    val_normal_files = normal_files[train_data_size:]
    val_abnormal_files = abnormal_files[train_data_size:]
    train_normal_files = normal_files[:train_data_size]
    train_abnormal_files = abnormal_files[:train_data_size]
    
    data_file_names = {
            "val_normal": val_normal_files,
            "val_abnormal": val_abnormal_files,
            "train_normal": train_normal_files,
            "train_abnormal": train_abnormal_files,
            "test_abnormal": test_abnormal_files,
            "test_normal": test_normal_files
            }

    return data_file_names


def read_data_arrays(data_file_names):

    train_normal_files = data_file_names["train_normal"]
    train_abnormal_files = data_file_names["train_abnormal"]
    val_normal_files = data_file_names["val_normal"]
    val_abnormal_files = data_file_names["val_abnormal"]
    test_normal_files = data_file_names["test_normal"]
    test_abnormal_files = data_file_names["test_abnormal"]

    train_normal_features=[np.load(f) for f in train_normal_files]
    train_abnormal_features=[np.load(f) for f in train_abnormal_files]
    val_normal_features=[np.load(f) for f in val_normal_files]
    val_abnormal_features=[np.load(f) for f in val_abnormal_files]
    test_normal_features=[np.load(f) for f in test_normal_files]
    test_abnormal_features=[np.load(f) for f in test_abnormal_files]

    train_normal_labels=[len(i)*[0] for i in train_normal_features]
    train_abnormal_labels=[len(i)*[1] for i in train_abnormal_features]
    val_normal_labels=[len(i)*[0] for i in val_normal_features]
    val_abnormal_labels=[len(i)*[1] for i in val_abnormal_features]
    test_normal_labels=[len(i)*[0] for i in test_normal_features]
    test_abnormal_labels=[len(i)*[1] for i in test_abnormal_features]

    train_features = train_normal_features + train_abnormal_features
    train_labels = train_normal_labels + train_abnormal_labels
    val_features = val_normal_features + val_abnormal_features
    val_labels = val_normal_labels + val_abnormal_labels
    test_features = test_normal_features + test_abnormal_features
    test_labels = test_normal_labels + test_abnormal_labels

    del train_normal_features
    del train_abnormal_features

    train_features = np.vstack(train_features)
    train_labels = np.hstack(train_labels)
    val_features = np.vstack(val_features)
    val_labels = np.hstack(val_labels)
    test_features = np.vstack(test_features)
    test_labels = np.hstack(test_labels)

    return (train_features, val_features, test_features,
            train_labels, val_labels, test_labels)



def standardize_data(train_features, val_features, test_features):
    scaler = StandardScaler3D()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.fit_transform(val_features)
    test_features = scaler.transform(test_features)
    return train_features, val_features, test_features


def data_loader(features, labels, batch_size):
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    features = torch.Tensor(features).float().to(device)
    labels = torch.Tensor(labels).float().to(device)
    data = torch.utils.data.TensorDataset(features, labels)
    data_iter = torch.utils.data.DataLoader(data, batch_size, shuffle=True)

    return data_iter, device


if __name__ == '__main__':
    print("Reading Data")
    data_file_names = data_file_names()
    (train_features, val_features, test_features,
     train_labels, val_labels, test_labels) = read_data_arrays(
             data_file_names)
    print("Scaling Data")
    train_features, val_features, test_features = standardize_data(
            train_features, val_features, test_features)
