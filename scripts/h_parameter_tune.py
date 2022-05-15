from matplotlib import tri
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
import argparse

def process_training_trial(trial: optuna.Trial):

    batch_size = trial.suggest_int("batch_size", 8, 32)
    hidden_size = trial.suggest_int("hidden_size", 100, 786)
    dropout = trial.suggest_float("dropout", 0.2, 0.9)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 6)
    lr = trial.suggest_float("lr", 0.0000001, 0.1)
    attn_layers = trial.suggest_int("attn_layers", 4, 15)
    out_layers = trial.suggest_int("out_layers", 4, 15)

    train_dataset = NewsDataset(prefix="nyt_all_", is_regression=IS_REGRESSION)
    val_dataset = NewsDataset(split='val', prefix="nyt_all_", is_regression=IS_REGRESSION)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Model(
        device,
        attn_layers=attn_layers,
        lstm_layers=lstm_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        out_layers=out_layers
    )

    results = train(
        train_dataset,
        val_dataset,
        model,
        device,
        batch_size=batch_size,
        max_epochs=80,
        lr=lr,
        is_regression=IS_REGRESSION
    )
    if device == "cuda:0":
        torch.cuda.empty_cache()

    if IS_REGRESSION:
        final_result = results["val_loss"][-1]
    else:
        final_result = results["val_accuracy"][-1]

    return final_result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--regression", type=bool, default=False, help="true if this is a regression problem")

    args = parser.parse_args()

    global IS_REGRESSION

    IS_REGRESSION = args.regression

    if IS_REGRESSION:
        study = optuna.create_study(direction="minimize")
    else:
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