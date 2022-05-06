import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def train(
    train_dataset,
    val_dataset,
    model,
    device,
    batch_size=32,
    max_epochs=1,
    lr=0.01,
    patience=5
):
    model.to(device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    results = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    last_loss = 10000.0
    trigger_times = 0

    for epoch in range(max_epochs):
        results["epoch"].append(epoch)

        train_running_loss = []
        train_running_accuracy = []

        model = model.train()
        for _, (x, y_true) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred, _ = model(x.float())
            y_true = y_true.reshape((-1, 1))
            loss = criterion(y_pred, y_true.float())

            loss.backward()
            optimizer.step()

            train_running_loss.append(loss.item())

            pred = np.round(y_pred.cpu().detach())
            target = np.round(y_true.cpu().detach())
            accuracy = accuracy_score(target, pred)
            train_running_accuracy.append(accuracy)

        train_loss = np.mean(train_running_loss)
        results["train_loss"].append(train_loss)
        train_accuracy = np.mean(train_running_accuracy)
        results["train_accuracy"].append(train_accuracy)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
        )

        val_running_accuracy = []
        val_running_loss = []
        model = model.eval()
        with torch.no_grad():

            for _, (x, y_true) in enumerate(val_dataloader):
                x = x.to(device)
                y_true = y_true.to(device)
                y_pred, _ = model(x.float())
                y_true = y_true.reshape((-1, 1))

                val_loss = criterion(y_pred, y_true.float())
                val_running_loss.append(val_loss.cpu().detach())

                pred = np.round(y_pred.cpu().detach())
                target = np.round(y_true.cpu().detach())
                accuracy = accuracy_score(target, pred)
                val_running_accuracy.append(accuracy)
        
        val_accuracy = np.mean(val_running_accuracy)
        results["val_accuracy"].append(val_accuracy)
        print({ 'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy })

        val_loss = np.mean(val_running_loss)

        if np.round(val_loss, 4) >= np.round(last_loss, 4):
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!')
                return results

        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = val_loss

    return results