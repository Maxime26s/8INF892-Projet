import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for data in tqdm(
        train_loader,
        desc="Training",
        leave=False,
        unit="batch",
        ncols=100,
    ):  # Wrap train_loader with tqdm
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
        for data in tqdm(
            data_loader,
            desc="Evaluating",
            leave=False,
            unit="batch",
            ncols=100,
        ):  # Wrap data_loader with tqdm
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(data_loader.dataset)


def train(
    model, train_loader, val_loader, optimizer, criterion, num_epochs, patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        val_loss = evaluate(model, criterion, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    return history
