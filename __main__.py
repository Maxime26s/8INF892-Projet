import argparse
from datetime import datetime
import logging
import torch
import numpy as np
import random
import os


def setup_logging():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_{current_time}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


setup_logging()
logger = logging.getLogger(__name__)

from src.data_loader import load_data
from src.hyperparameter_tuning import grid_search, generate_param_grid
from src.gcn_model import GCN
from src.gat_model import GAT
from src.gsage_model import GraphSAGE
from src.train import train
from src.visualizer import visualize_history


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module
    torch.manual_seed(seed)  # PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch's CUDA seed
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU, for all CUDA devices

    # Additional configurations for ensuring reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setting environment variables for deterministic behavior in PyTorch < 1.8
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # or 'CUBLAS_WORKSPACE_CONFIG=:16:8' for older versions
    )

    logging.info(f"Seed set to: {seed}")


if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logging.info("CUDA is not available, using CPU instead")


def training(model_type="gcn"):
    logger.info("Starting training")

    train_loader, val_loader, _ = load_data(batch_size=32)

    # Initialize the model and optimizer
    if model_type == "gcn":
        model = GCN(
            in_channels=train_loader.dataset.num_features,
            out_channels=train_loader.dataset.num_tasks,
            hidden_channels=256,
            num_layers=5,
            dropout=0.2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif model_type == "gsage":
        model = GraphSAGE(
            in_channels=train_loader.dataset.num_features,
            out_channels=train_loader.dataset.num_tasks,
            hidden_channels=256,
            num_layers=5,
            dropout=0.2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif model_type == "gat":
        model = GAT(
            in_channels=train_loader.dataset.num_features,
            out_channels=train_loader.dataset.num_tasks,
            hidden_channels=64,
            num_layers=5,
            dropout=0.2,
            heads=8,
            concat=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        logger.error(
            "Invalid model type. Please choose either 'gcn', 'gat', or 'gsage'."
        )
        raise ValueError("Invalid model type specified.")

    logger.info(f"Model initialized: {model}")

    criterion = torch.nn.BCEWithLogitsLoss()

    # Train the model
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=300,
        patience=None,
    )

    # Visualize the training history
    visualize_history(history, "loss", f"Training and Validation", f"history_loss.png")
    visualize_history(history, "acc", f"Training and Validation", f"history_acc.png")
    visualize_history(history, "roc_auc", f"Validation", f"history_roc-auc.png")

    logger.info("Training completed")


def hyperparameter_tuning(model_type="gcn"):
    logger.info("Starting hyperparameter tuning")

    if model_type == "gcn":
        model_class = GCN
    elif model_type == "gat":
        model_class = GAT
    elif model_type == "gsage":
        model_class = GraphSAGE
    else:
        logger.error(
            "Invalid model type. Please choose either 'gcn', 'gat', or 'gsage'."
        )
        raise ValueError("Invalid model type specified.")

    train_loader, val_loader, _ = load_data(batch_size=32)

    # Define the hyperparameter grid for the model
    if model_class == GAT:
        param_options = {
            "hidden_channels": [64, 256],
            "num_layers": [3, 5],
            "dropout": [0.2],
            "lr": [0.001, 0.0001],
            "heads": [2, 4, 8],
            "concat": [True, False],
        }
    else:
        param_options = {
            "hidden_channels": [64, 256],
            "num_layers": [3, 5],
            "dropout": [0.2, 0.5],
            "lr": [0.001, 0.0001],
        }

    # Generate a grid of hyperparameters
    param_grid = generate_param_grid(param_options)
    _, _, results = grid_search(
        model_class, train_loader, val_loader, param_grid, num_epochs=100, patience=15
    )

    # Sort the results by the maximum ROC-AUC
    sorted_results = sorted(
        results, key=lambda result: result["max_roc_auc"], reverse=True
    )

    # Print the top 5 hyperparameter sets by ROC-AUC
    logger.info("Top 5 Hyperparameter Sets by ROC-AUC:")
    for i, result in enumerate(sorted_results[:5]):
        logger.info(
            f"Rank {i+1}: ROC-AUC: {result['max_roc_auc']:.4f}, Parameters: {result['params']}"
        )

    # Visualize the training history for the top 5 hyperparameter sets
    for index, result in enumerate(sorted_results, start=1):
        history = result["history"]
        params = result["params"]

        visualize_history(
            history, "loss", f"Params: {params}", f"history_loss_{index}.png"
        )
        visualize_history(
            history, "acc", f"Params: {params}", f"history_acc_{index}.png"
        )
        visualize_history(
            history, "roc_auc", f"Params: {params}", f"history_roc-auc_{index}.png"
        )

    logger.info("Hyperparameter tuning completed")


if __name__ == "__main__":
    set_seed(42)

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
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "gat", "gsage"],
        required=True,
        help="Specify the model type to use: 'gcn', 'gat', or 'gsage'.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        training()
    elif args.mode == "tune":
        hyperparameter_tuning()
    else:
        logger.error("Invalid mode. Please choose either 'train' or 'tune'.")
        raise ValueError("Invalid mode specified.")

    logger.info("Application finished successfully")
