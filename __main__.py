import argparse
from datetime import datetime
import logging
import torch
import torch.nn.functional as F
import numpy as np
import random
import os


def setup_logging():
    # Get current time for the log file name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_{current_time}.log"

    # Set the root logger's level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # Create a file handler with a timestamped log file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


setup_logging()
logger = logging.getLogger(__name__)

from src.data_loader import load_data
from src.hyperparameter_tuning import grid_search, generate_param_grid
from src.gcn_model import GCN
from src.train import train
from src.visualizer import visualize_history, visualize_all_histories


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

    logging.info(f"Seed set to: {seed}")


if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logging.info("CUDA is not available, using CPU instead")


def training():
    logger.info("Starting training")

    train_loader, val_loader, _ = load_data(batch_size=64)  # Define your data loader
    model = GCN(
        num_features=train_loader.dataset.num_features,
        num_labels=train_loader.dataset.num_classes,
        hidden_channels=64,
        num_layers=2,
        dropout=0.5,
    )
    logger.info(f"Model initialized: {model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = F.mse_loss

    # Train the model
    history = train(
        model, train_loader, val_loader, optimizer, criterion, num_epochs=20
    )

    visualize_history(history, filename="training_history.png")

    logger.info("Training completed")


def hyperparameter_tuning():
    logger.info("Starting hyperparameter tuning")

    train_loader, val_loader, _ = load_data(batch_size=64)  # Define your data loader
    param_options = {
        "hidden_channels": [16, 32, 64],
        "num_layers": [2, 3, 4],
        "dropout": [0.1, 0.25, 0.5],
        "lr": [0.01, 0.001, 0.0001],
    }

    param_grid = generate_param_grid(param_options)
    best_params, best_val_loss, histories = grid_search(
        GCN, train_loader, val_loader, param_grid, num_epochs=20
    )

    logger.info(f"Best Hyperparameters: {best_params}")
    logger.info(f"Best Validation Loss: {best_val_loss}")

    visualize_all_histories(histories)

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

    # Parse the arguments
    args = parser.parse_args()

    if args.mode == "train":
        training()
    elif args.mode == "tune":
        hyperparameter_tuning()
    else:
        logger.error("Invalid mode. Please choose either 'train' or 'tune'.")
        raise ValueError("Invalid mode specified.")

    logger.info("Application finished successfully")
