import logging
import torch

logger = logging.getLogger(__name__)


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)


def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(data_loader.dataset)


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        val_loss = evaluate(model, criterion, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    return history
