import itertools
import logging
import torch
from .train import train

logger = logging.getLogger(__name__)


def grid_search(
    model_class, train_loader, val_loader, param_grid, num_epochs, patience
):
    logger.info("Starting hyperparameter grid search")

    best_roc_auc = 0
    best_params = None
    results = []

    for index, params in enumerate(param_grid, start=1):
        logger.info(f"Testing combination {index}/{len(param_grid)}: {params}")

        # Separate model and optimizer parameters
        model_params = {k: v for k, v in params.items() if k != "lr"}
        lr = params.get("lr", 0.001)  # Default learning rate if not specified

        # Initialize model with the current set of hyperparameters
        model = model_class(
            in_channels=train_loader.dataset.num_features,
            out_channels=train_loader.dataset.num_tasks,
            **model_params,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Train the model with the current hyperparameter configuration
        history = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            num_epochs,
            patience,
        )

        max_roc_auc = max(history["roc_auc"])

        # Update best params if current configuration is better
        if max_roc_auc > best_roc_auc:
            best_roc_auc = max_roc_auc
            best_params = params

        # Store the history for the current parameter combination
        results.append(
            {"params": params, "history": history, "max_roc_auc": max_roc_auc}
        )

    logger.info(f"Best ROC-AUC: {best_roc_auc:.4f} with parameters {best_params}")
    logger.info("Hyperparameter grid search completed")
    return best_params, best_roc_auc, results


def generate_param_grid(param_options):
    logger.info("Generating parameter grid")

    param_names = param_options.keys()
    param_values = param_options.values()

    # Generate all combinations of hyperparameter values
    all_combinations = list(itertools.product(*param_values))
    param_grid = [
        dict(zip(param_names, combination)) for combination in all_combinations
    ]

    logger.info("Parameter grid generation completed")
    return param_grid
