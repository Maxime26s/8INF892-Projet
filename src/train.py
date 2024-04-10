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

    for data in tqdm(
        train_loader,
        desc="Training",
        leave=False,
        unit="batch",
        ncols=100,
    ):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)

        target = data.y.squeeze(1)

        loss = criterion(out, target)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

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
        for data in tqdm(
            data_loader,
            desc="Evaluating",
            leave=False,
            unit="batch",
            ncols=100,
        ):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)

            target = data.y.squeeze(1)

            loss = criterion(out, target)
            total_loss += loss.item()

            _, predicted = torch.max(out, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            probs = torch.sigmoid(
                out[:, 1]
            )  # Assuming out[:, 1] is the output for class 1
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    logger.info("Confusion Matrix:\n" + np.array2string(conf_matrix))

    roc_auc = roc_auc_score(all_targets, all_probs)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

    # Print confusion matrix

    # Calculate and print class distribution
    class_distribution = np.bincount(all_targets)
    class_distribution_str = ", ".join(
        f"Class {i}: {n}" for i, n in enumerate(class_distribution)
    )
    logger.info("Class Distribution: " + class_distribution_str)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train(
    model, train_loader, val_loader, optimizer, criterion, num_epochs, patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, optimizer, criterion, train_loader, device
        )
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%"
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
