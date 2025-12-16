# plotting.py
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_confusion_matrix(
    cm,
    class_names,
    title,
    save_path=None,
    cmap="Blues"
):
    """
    Plota uma confusion matrix (já agregada ou de um fold).
    """

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def plot_model_comparison(
    df_summary,
    save_path=None
):
    """
    Plota comparação entre modelos (Accuracy e F1).
    """

    x = np.arange(len(df_summary))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, df_summary["MeanAcc"], width, label="Mean Accuracy")
    plt.bar(x + width / 2, df_summary["MeanF1"], width, label="Mean F1")

    plt.xticks(x, df_summary["Model"])
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close()
    #plt.show()
