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
 
if __name__ == '__main__':
 
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--plot", type=str, default="plot", help="name of plot file")
    parser.add_argument("-A", "--attention", type=bool, default="False", help="true or false to add attention layer")
    parser.add_argument("-B", "--batch", type=int, default=32, help="batch size")
    parser.add_argument("-E", "--epochs", type=int, default=75, help="max number of epochs")
    parser.add_argument("-H", "--hidden_size", type=int, default=384, help="LSTM layer output size")
    parser.add_argument("-D", "--dropout", type=float, default=0.7, help="dropout rate")
    parser.add_argument("-R", "--prefix", type=str, default='', help="prefix in data file names")
    parser.add_argument("-L", "--lstm_layers", type=int, default=1, help="number of lstm layers")
    
    # Read arguments from command line
    args = parser.parse_args()

    batch_size = args.batch
    max_epochs = args.epochs
    hidden_size = args.hidden_size
    dropout = args.dropout
    plot_file_name = args.plot
    prefix = args.prefix
    lstm_layers = args.lstm_layers

    train_dataset = NewsDataset(prefix=prefix)
    val_dataset = NewsDataset(split='val', prefix=prefix)
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
        max_epochs=max_epochs
    )
    if device == "cuda:0":
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df = results_df.drop(columns=["train_loss"])
    sns.lineplot(x="epoch", y="value", hue="variable", data=pd.melt(results_df, ["epoch"]))
    current_dir = pathlib.Path(__file__).parent.resolve()
    plt.savefig(f'{current_dir}/../plots/{plot_file_name}.png', dpi=300)
