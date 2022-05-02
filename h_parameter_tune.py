import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import argparse
from scripts.datasets import NewsDataset
from scripts.model import Model
from scripts.training import train
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import optuna


def process_training_trial(trial: optuna.Trial):

    batch_size = trial.suggest_int("batch_size", 8, 32)
    hidden_size = trial.suggest_int("hidden_size", 100, 786)
    dropout = trial.suggest_float("dropout", 0.2, 0.9)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 4)

    train_dataset = NewsDataset(prefix="nyt_")
    val_dataset = NewsDataset(split='val', prefix="nyt_")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Model(
        device,
        lstm_layers=lstm_layers,
        hidden_size=hidden_size,
        dropout=dropout
    )

    results = train(
        train_dataset,
        val_dataset,
        model,
        device,
        batch_size=batch_size,
        max_epochs=80
    )
    if device == "cuda:0":
        torch.cuda.empty_cache()

    final_accuracy = results["val_accuracy"][-1]

    return final_accuracy

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(process_training_trial, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))