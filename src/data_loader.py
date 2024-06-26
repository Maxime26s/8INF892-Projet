import logging
from ogb.graphproppred import PygGraphPropPredDataset
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
):
    transform = Compose([Distance(norm=False, max_value=None)])

    # Load the dataset
    dataset = PygGraphPropPredDataset(root="/tmp/ogbg-molhiv", name="ogbg-molhiv")
    print(dataset.num_tasks)
    dataset = dataset.shuffle()

    # Split the dataset into training, validation, and test sets
    num_graphs = len(dataset)
    train_size = int(num_graphs * train_prop)
    val_size = int(num_graphs * val_prop)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size :]

    # Initialize the DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.debug("Datasets prepared and loaders initialized")

    return train_loader, val_loader, test_loader
