import torch
from tqdm import tqdm
import torch.nn as nn

from load_data import read_data_arrays, data_file_names, standardize_data, data_loader
from models import ChronoNet
from utils import cal_accuracy, evaluate_model 


BATCH_SIZE = 128

def main():
    print("Reading Data....")
    data_files = data_file_names()
    (train_features, val_features, test_features,
     train_labels, val_labels, test_labels) = read_data_arrays(
             data_files)
    
    print("Scaling Data....")
    train_features, val_features, test_features = standardize_data(
            train_features, val_features, test_features)
    
    print("Data Loader....")
    train_iter, device = data_loader(train_features, train_labels, BATCH_SIZE)
    val_iter, device = data_loader(val_features, val_labels, BATCH_SIZE)
    test_iter, device = data_loader(test_features, test_labels, BATCH_SIZE)
    
    print("Training Model....")
    n_chans = 19
    model=ChronoNet(n_chans)
    model.to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    for epoch in range(1, epochs + 1):
        print("Epoch", epoch) 
        loss_sum, n = 0.0, 0
        model.train()
        for t, (x, y) in enumerate(tqdm(train_iter)):
            y_pred = model(x)
            y_pred = y_pred.squeeze()
            loss = loss_func(y_pred, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            optimizer.zero_grad()
    
        val_loss = evaluate_model(model, loss_func, val_iter)
        print("Train loss:", loss_sum / (t+1), "Accuracy: ", 
            cal_accuracy(model, train_labels, train_features)[0])
        print("Val loss:", val_loss, ", Accuracy: ", 
            cal_accuracy(model, val_labels, val_features)[0])

if __name__ == '__main__':
    main()
