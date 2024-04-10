import logging
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import QM9, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, Distance
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_data(
    batch_size=32,
    train_prop=0.7,
    val_prop=0.2,
    max_graph_count=None,
):
    transform = Compose([Distance(norm=False, max_value=None)])

    # Load the dataset
    dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")

    # Select a subset of the first 5000 compounds
    if max_graph_count is not None:
        subset_size = min(max_graph_count, len(dataset))
        dataset = dataset[:subset_size]

    dataset = dataset.shuffle()

    # Print information about the dataset
    print_info(dataset, "Original")

    # Split the dataset into training, validation, and test sets
    num_graphs = len(dataset)
    train_size = int(num_graphs * train_prop)
    val_size = int(num_graphs * val_prop)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size :]

    # Print information about the splits
    print_info(train_dataset, "Train")
    print_info(val_dataset, "Validation")
    print_info(test_dataset, "Test")

    # Initialize the DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.debug("Datasets prepared and loaders initialized")

    return train_loader, val_loader, test_loader


def print_info(dataset, name):
    logger.info(f"{name} - Dataset type: {type(dataset)}")
    logger.info(f"{name} - Number of graphs: {len(dataset)}")
    logger.info(f"{name} - Number of features: {dataset.num_features}")
    logger.info(f"{name} - Number of classes: {dataset.num_classes}")
    logger.info(f"{name} - Number of nodes: {sum(data.num_nodes for data in dataset)}")
    logger.info(f"{name} - Number of features per node: {dataset.num_node_features}")
    logger.info(f"{name} - Number of edges: {sum(data.num_edges for data in dataset)}")
    logger.info(f"{name} - Number of features per edge: {dataset.num_edge_features}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    train_loader, val_loader, test_loader = load_data(max_graph_count=None)
