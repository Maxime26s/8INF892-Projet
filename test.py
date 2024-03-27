import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """Sets the seed for reproducibility across multiple packages."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch CUDA
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA (all GPUs)

    # Ensuring deterministic behavior in PyTorch (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables for further ensuring deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # For PyTorch < 1.8, you can use 'CUBLAS_WORKSPACE_CONFIG=:16:8'
    )


set_seed(42)  # Set a seed value

# Step 1: Load the QM9 dataset
dataset = QM9(root="/tmp/QM9")
dataset = dataset.shuffle()

# Split dataset into train, validation, and test
train_dataset = dataset[:10000]
val_dataset = dataset[10000:12000]
test_dataset = dataset[12000:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Step 2: Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 19)  # Predicting a single property

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First Graph Convolution layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second Graph Convolution layer
        x = self.conv2(x, edge_index)

        # Global Mean Pooling
        x = global_mean_pool(x, batch)

        # Output layer
        x = self.out(x)

        return x


model = GCN(hidden_channels=64)
print(model)

# Step 3: Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Step 4: Define the training loop
def train():
    model.train()

    for data in train_loader:
        optimizer.zero_grad()  # Clear gradients
        out = model(data)  # Forward pass
        loss = F.mse_loss(out, data.y)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters


# Step 5: Define the test function
def test(loader):
    model.eval()
    error = 0

    for data in loader:
        with torch.no_grad():
            pred = model(data)
            error += (pred - data.y).abs().sum().item()  # Sum up batch error

    return error / len(loader.dataset)


# Step 6: Run the training and evaluation
for epoch in range(1, 201):
    train()
    train_error = test(train_loader)
    val_error = test(val_loader)
    print(
        f"Epoch: {epoch:03d}, Train Error: {train_error:.4f}, Val Error: {val_error:.4f}"
    )

# Test the model
test_error = test(test_loader)
print(f"Test Error: {test_error:.4f}")
