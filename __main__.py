import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from src.data_loader import load_data
from src.hyperparameter_tuning import grid_search, generate_param_grid
from src.gcn_model import GCN
from src.train import train

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module
    torch.manual_seed(seed)  # PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch's CUDA seed
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU, for all CUDA devices

    # Additional configurations for further ensuring reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setting environment variables for deterministic behavior in PyTorch < 1.8
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # or 'CUBLAS_WORKSPACE_CONFIG=:16:8' for older versions
    )


set_seed(42)


def training():
    train_loader, val_loader, _ = load_data(batch_size=64)  # Define your data loader
    model = GCN(
        num_features=train_loader.dataset.num_features,
        num_labels=train_loader.dataset.num_classes,
        hidden_channels=64,
        num_layers=2,
        dropout=0.5,
    )
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = F.mse_loss

    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, num_epochs=20)


def hyperparameter_tuning():
    train_loader, val_loader, _ = load_data(batch_size=64)  # Define your data loader
    param_options = {
        "hidden_channels": [16, 32, 64],
        "num_layers": [2, 3, 4],
        "dropout": [0.1, 0.25, 0.5],
        "lr": [0.01, 0.001, 0.0001],
    }

    param_grid = generate_param_grid(param_options)
    best_params, best_val_loss = grid_search(
        GCN, train_loader, val_loader, param_grid, num_epochs=20
    )
    print("Best Hyperparameters:", best_params)
    print("Best Validation Loss:", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GCN model training or hyperparameter grid search."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "tune"],
        required=True,
        help="Specify the mode to run: 'train' or 'tune'.",
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.mode == "train":
        training()
    elif args.mode == "tune":
        hyperparameter_tuning()
    else:
        print("Invalid mode. Please choose either 'train' or 'tune'.")
