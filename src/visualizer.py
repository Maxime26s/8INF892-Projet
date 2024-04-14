import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_history(history, metric, title="Training and Validation", filename=None):
    plt.figure(figsize=(10, 6))

    if metric != "roc_auc":
        train_metric = f"train_{metric}"
        val_metric = f"val_{metric}"

        train_values = to_numpy(history[train_metric])
        val_values = to_numpy(history[val_metric])

        plt.plot(train_values, label=f"Train {metric.capitalize()}")
        plt.plot(val_values, label=f"Validation {metric.capitalize()}")
    else:
        values = to_numpy(history[metric])

        plt.plot(values, label=f"{metric.capitalize()}")

    plt.title(f"{title} - {metric.capitalize()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def to_numpy(data):
    if torch.is_tensor(data[0]):
        return np.array([d.cpu().numpy() for d in data])
    else:
        return np.array(data)
