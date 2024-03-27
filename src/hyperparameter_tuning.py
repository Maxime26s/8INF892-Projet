import itertools
import logging
import torch
from .train import train

logger = logging.getLogger(__name__)


def grid_search(model_class, train_loader, val_loader, param_grid, num_epochs):
    logger.info("Starting hyperparameter grid search")

    best_val_loss = float("inf")
    best_params = None
    histories = {}  # Dictionary to store history for each parameter combination

    for index, params in enumerate(param_grid, start=1):
        logger.info(f"Testing combination {index}/{len(param_grid)}: {params}")

        # Separate model and optimizer parameters
        model_params = {k: v for k, v in params.items() if k != "lr"}
        lr = params.get("lr", 0.001)  # Default learning rate if not specified

        # Initialize model with the current set of hyperparameters
        model = model_class(
            num_features=train_loader.dataset.num_features,
            num_labels=train_loader.dataset.num_classes,
            **model_params,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        # Train the model with the current hyperparameter configuration
        history = train(
            model, train_loader, val_loader, optimizer, criterion, num_epochs
        )

        # Use the best validation loss as the performance metric
        final_val_loss = history["val_loss"][-1]

        logger.info(f"Params: {params}, Final Val Loss: {final_val_loss:.4f}")

        # Update best params if current configuration is better
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = params

        # Store the history for the current parameter combination
        histories[tuple(params.items())] = history

    logger.info("Hyperparameter grid search completed")
    return best_params, best_val_loss, histories


def generate_param_grid(param_options):
    logger.info("Generating parameter grid")

    # Extract parameter names and their corresponding lists of possible values
    param_names = param_options.keys()
    param_values = param_options.values()

    # Generate all combinations of hyperparameter values
    all_combinations = list(itertools.product(*param_values))

    # Convert each combination to a dictionary
    param_grid = [
        dict(zip(param_names, combination)) for combination in all_combinations
    ]

    logger.info("Parameter grid generation completed")
    return param_grid
