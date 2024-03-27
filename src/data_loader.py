import logging
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, Distance
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_data(batch_size=32, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    transform = Compose([Distance()])

    dataset = QM9(root="/tmp/QM9", transform=transform)

    logger.info(f"Total number of graphs in the dataset: {len(dataset)}")
    logger.info(f"Number of features per node: {dataset.num_node_features}")
    logger.info(f"Number of features per edge: {dataset.num_edge_features}")
    logger.info(f"Number of graph labels: {dataset.num_classes}")

    if not all(
        data.is_undirected()
        for data in tqdm(
            dataset,
            desc="Sanity Check: No directed graphs",
            unit="graph",
            ncols=100,
        )
    ):
        raise ValueError("All graphs in the dataset should be undirected.")

    dataset = dataset.shuffle()

    # Stack all targets
    all_targets = torch.stack(
        [
            data.y
            for data in tqdm(dataset, desc="Stacking Targets", unit="graph", ncols=100)
        ],
        dim=0,
    )

    # Calculate mean and std dev for each of the 19 targets
    target_means = all_targets.mean(dim=0)
    target_stds = all_targets.std(dim=0)

    if not all(
        target_std == 0
        for target_std in tqdm(
            target_stds,
            desc="Sanity Check: Zero Standard Deviation",
            unit="target",
            ncols=100,
        )
    ):
        raise ValueError(
            "Standard deviation of a target is zero, which will lead to division by zero during standardization."
        )

    # Standardize the targets
    for data in tqdm(dataset, desc="Standardizing Targets", unit="graph", ncols=100):
        data.y = (data.y - target_means) / target_stds

    if not all(
        torch.isfinite(data.y).all()
        for data in tqdm(
            dataset,
            desc="Sanity Check: Non-finite Values",
            unit="graph",
            ncols=100,
        )
    ):
        raise ValueError("Non-finite values found in standardized targets.")

    # Calculate means and std devs for node features
    concatenated_features = torch.cat(
        [
            data.x
            for data in tqdm(
                dataset, desc="Concatenating Features", unit="graph", ncols=100
            )
        ],
        dim=0,
    )
    feature_means = torch.mean(concatenated_features, dim=0)
    feature_stds = torch.std(concatenated_features, dim=0)

    # Standardize node features
    for data in tqdm(dataset, desc="Standardizing Features", unit="graph", ncols=100):
        data.x = (data.x - feature_means) / feature_stds

    if not all(
        torch.isfinite(data.x).all()
        for data in tqdm(
            dataset,
            desc="Sanity Check: Non-finite Values",
            unit="graph",
            ncols=100,
        )
    ):
        raise ValueError("Non-finite values found in standardized features.")

    # Replace NaNs and Infs (post-standardization)
    for data in tqdm(dataset, desc="Handling Missing Data", unit="graph", ncols=100):
        data.x = torch.nan_to_num(data.x, nan=0.0)
        data.y = torch.nan_to_num(data.y, nan=0.0)

    # Ensure that the proportions sum up to 1
    total = train_prop + val_prop + test_prop
    assert abs(total - 1.0) < 1e-6, f"Proportions must sum up to 1, but sum is {total}."

    # Calculate split indices
    total_size = len(dataset)
    train_end = int(train_prop * total_size)
    val_end = train_end + int(val_prop * total_size)

    # Split the dataset based on calculated indices
    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]

    if not (len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset)):
        raise ValueError("Sum of split datasets does not equal total dataset size.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.debug("Datasets prepared and loaders initialized")

    return train_loader, val_loader, test_loader
