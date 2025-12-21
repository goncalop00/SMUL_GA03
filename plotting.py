# plotting.py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import itertools  # Added for looping over matrix cells

matplotlib.use("Agg")  # non-interactive backend

def plot_confusion_matrix(
    cm,
    class_names,
    title,
    save_path=None,
    cmap="Blues",
    normalize=True
):
    """
    Plots a confusion matrix.
    If normalize=True, converts counts to percentages (Recall).
    """
    
    # 1. Normalize the matrix if requested
    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1 
        cm_plot = cm.astype('float') / row_sums
        fmt = '.2f'
        title = f"{title} (Normalized)"
    else:
        cm_plot = cm
        fmt = 'd'

    plt.figure(figsize=(10, 8)) # Slightly larger for text
    plt.imshow(cm_plot, interpolation="nearest", cmap=cmap, vmin=0, vmax=1 if normalize else None)
    plt.title(title, fontsize=14)
    plt.colorbar(label="Accuracy (Recall)" if normalize else "Count")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # 2. Add Text Annotations inside the squares
    thresh = cm_plot.max() / 2.
    for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
        val = cm_plot[i, j]
        plt.text(j, i, format(val, fmt),
                 horizontalalignment="center",
                 color="white" if val > thresh else "black")

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300) # Higher DPI for report quality

    plt.close()


def plot_model_comparison(
    df_summary,
    save_path=None
):
    """
    Plota comparação entre modelos (Accuracy e F1).
    """

    x = np.arange(len(df_summary))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, df_summary["MeanAcc"], width, label="Mean Accuracy", color="#4c72b0")
    plt.bar(x + width / 2, df_summary["MeanF1"], width, label="Mean F1", color="#55a868")

    plt.xticks(x, df_summary["Model"])
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend(loc="lower right")
    plt.ylim(0, 1.0) # Fix y-axis from 0 to 1 for easier reading

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.close()