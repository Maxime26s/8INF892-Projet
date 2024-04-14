import logging
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(
        train_loader,
        desc="Training",
        leave=False,
        unit="batch",
        ncols=100,
    ):
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.batch)
        optimizer.zero_grad()

        loss = criterion(out.to(torch.float32), batch.y.to(torch.float32))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(out).squeeze()
        predicted = (probs > 0.5).long()

        total += batch.y.size(0)
        correct += (predicted == batch.y.squeeze(1)).sum()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Evaluating",
            leave=False,
            unit="batch",
            ncols=100,
        ):
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y.to(torch.float32))
            total_loss += loss.item()

            probs = torch.sigmoid(out).squeeze()
            predicted = (probs > 0.5).long()

            total += batch.y.size(0)
            correct += (predicted == batch.y.squeeze(1)).sum()

            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_preds.extend(predicted.cpu().numpy().flatten().tolist())
            all_targets.extend(batch.y.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    roc_auc = roc_auc_score(all_targets, all_probs)

    return avg_loss, accuracy, roc_auc


def train(
    model, train_loader, val_loader, optimizer, criterion, num_epochs, patience=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "roc_auc": [],
        "max_roc_auc": 0,
    }
    best_roc_auc = 0
    best_epoch = 0
    last_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, optimizer, criterion, train_loader, device
        )
        val_loss, val_acc, roc_auc = evaluate(model, criterion, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["roc_auc"].append(roc_auc)

        logger.info(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%, ROC-AUC: {roc_auc:.4f}"
        )

        last_epoch = epoch

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_epoch = epoch
            patience_counter = 0
        elif patience is not None:
            patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    history["max_roc_auc"] = best_roc_auc
    last_roc_auc = history["roc_auc"][-1]

    logger.info(f"Best ROC-AUC: {best_roc_auc:.4f} at epoch {best_epoch}")
    logger.info(f"Final ROC-AUC: {last_roc_auc:.4f} at epoch {last_epoch}")

    return history
