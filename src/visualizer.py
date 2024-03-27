import matplotlib.pyplot as plt


def visualize_history(history, title="Training and Validation Loss", filename=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def visualize_all_histories(histories):
    for index, (params, history) in enumerate(histories.items(), start=1):
        title = f"Params: {dict(params)}"
        filename = f"history_plot_{index}.png"
        visualize_history(history, title, filename)
