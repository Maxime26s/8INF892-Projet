import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


def load_data(batch_size=32, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    dataset = QM9(root="/tmp/QM9")

    print(f"Total number of graphs in the dataset: {len(dataset)}")
    print(f"Number of features per node: {dataset.num_node_features}")
    print(f"Number of features per edge: {dataset.num_edge_features}")
    print(f"Number of graph labels: {dataset.num_classes}")

    dataset = dataset.shuffle()

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
