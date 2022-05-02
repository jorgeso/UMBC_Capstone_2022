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
    max_epochs=1
):
    model.to(device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    results = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

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

        model = model.eval()
        with torch.no_grad():

            for _, (x, y_true) in enumerate(val_dataloader):
                x = x.to(device)
                y_true = y_true.to(device)
                y_pred, _ = model(x.float())
                y_true = y_true.reshape((-1, 1))

                pred = np.round(y_pred.cpu().detach())
                target = np.round(y_true.cpu().detach())
                accuracy = accuracy_score(target, pred)
                val_running_accuracy.append(accuracy)
        
        val_accuracy = np.mean(val_running_accuracy)
        results["val_accuracy"].append(val_accuracy)
        print({ 'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy })

    return results