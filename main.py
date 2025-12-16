# main.py
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import (
    CONFIG,
    RUN_SINGLE_FOLD,
    SINGLE_FOLD,
    TUNE,
    print_experiment_config
)

from data import (
    load_dataset,
    build_metadata,
    build_fold_dataset
)

from features import extract_mfcc

from svm_model import tune_svm
from rf_model import tune_rf

from evaluation import evaluate
from plotting import plot_confusion_matrix, plot_model_comparison



# -----------------------------------------------------
# Setup
# -----------------------------------------------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print_experiment_config(CONFIG)

# -----------------------------------------------------
# Load dataset and metadata
# -----------------------------------------------------
print("Loading UrbanSound8K...")
dataset = load_dataset()
meta = build_metadata(dataset)

# -----------------------------------------------------
# Folds to run
# -----------------------------------------------------
if RUN_SINGLE_FOLD:
    folds_to_run = [SINGLE_FOLD]
else:
    folds_to_run = range(1, 11)

# -----------------------------------------------------
# Class names (for confusion matrices)
# -----------------------------------------------------
CLASS_NAMES = [
    "air_cond", "car_horn", "children", "dog_bark", "drilling",
    "engine", "gun_shot", "jackhammer", "siren", "street_music"
]

# -----------------------------------------------------
# Storage
# -----------------------------------------------------
all_results = {}
svm_params_per_fold = []
rf_params_per_fold = []

total_start = time.time()

# -----------------------------------------------------
# Run experiments
# -----------------------------------------------------
for model_name in ["SVM", "RandomForest"]:

    print("\n==============================================")
    print(f"        MODEL: {model_name}")
    print("==============================================")

    fold_metrics = []
    conf_matrices = []

    for fold in folds_to_run:
        print("\n----------------------------------------------")
        print(f"               TEST FOLD {fold} - {model_name}")
        print("----------------------------------------------")

        X_train, y_train, X_test, y_test = build_fold_dataset(
            meta,
            extract_mfcc,
            fold
        )

        # -------------------------
        # Train / Tune
        # -------------------------
        t0 = time.time()

        if model_name == "SVM":
            if TUNE:
                model, best_params = tune_svm(
                    X_train, y_train,
                    X_test, y_test
                )
                svm_params_per_fold.append({
                    "fold": fold,
                    "C": best_params[0],
                    "gamma": best_params[1]
                })
            else:
                raise NotImplementedError("Non-tuned SVM not implemented in modular version.")

        elif model_name == "RandomForest":
            if TUNE:
                model, best_params = tune_rf(
                    X_train, y_train,
                    X_test, y_test
                )
                rf_params_per_fold.append({
                    "fold": fold,
                    "n_estimators": best_params[0],
                    "max_depth": best_params[1],
                    "min_samples_split": best_params[2]
                })
            else:
                raise NotImplementedError("Non-tuned RF not implemented in modular version.")

        train_time = time.time() - t0

        # -------------------------
        # Evaluation
        # -------------------------
        acc, prec, rec, f1, preds, pred_time = evaluate(
            model,
            X_test,
            y_test
        )

        print(
            f"{model_name} Fold {fold} → "
            f"Accuracy: {acc:.3f}, "
            f"F1: {f1:.3f}, "
            f"train_time: {train_time:.1f}s, "
            f"pred_time: {pred_time:.2f}s"
        )

        fold_metrics.append([
            fold, acc, prec, rec, f1, train_time, pred_time
        ])

        cm = confusion_matrix(
            y_test,
            preds,
            labels=list(range(10))
        )
        conf_matrices.append(cm)

    # -------------------------------------------------
    # Aggregate results for this model
    # -------------------------------------------------
    df_metrics = pd.DataFrame(
        fold_metrics,
        columns=[
            "Fold", "Accuracy", "Precision",
            "Recall", "F1",
            "TrainTime_s", "PredTime_s"
        ]
    )

    mean_cm = np.mean(conf_matrices, axis=0)

    all_results[model_name] = {
        "df_metrics": df_metrics,
        "mean_confusion": mean_cm
    }

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    df_metrics.to_csv(
        f"{RESULTS_DIR}/fold_metrics_{model_name}.csv",
        index=False
    )

    np.savetxt(
        f"{RESULTS_DIR}/mean_confusion_matrix_{model_name}.csv",
        mean_cm,
        delimiter=","
    )

    plot_confusion_matrix(
        mean_cm,
        CLASS_NAMES,
        title=f"Mean Confusion Matrix ({model_name})",
        save_path=f"{RESULTS_DIR}/mean_confusion_matrix_{model_name}.png"
    )

# -----------------------------------------------------
# Save hyperparameters per fold
# -----------------------------------------------------
if TUNE:
    if svm_params_per_fold:
        pd.DataFrame(svm_params_per_fold).to_csv(
            f"{RESULTS_DIR}/svm_best_params_per_fold.csv",
            index=False
        )
    if rf_params_per_fold:
        pd.DataFrame(rf_params_per_fold).to_csv(
            f"{RESULTS_DIR}/rf_best_params_per_fold.csv",
            index=False
        )

# -----------------------------------------------------
# Summary comparison
# -----------------------------------------------------
summary_rows = []
for model_name, res in all_results.items():
    df = res["df_metrics"]
    summary_rows.append({
        "Model": model_name,
        "MeanAcc": df["Accuracy"].mean(),
        "MeanF1": df["F1"].mean(),
        "MeanTrain_s": df["TrainTime_s"].mean(),
        "MeanPred_s": df["PredTime_s"].mean(),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(
    f"{RESULTS_DIR}/comparison_summary.csv",
    index=False
)

plot_model_comparison(
    df_summary,
    save_path=f"{RESULTS_DIR}/comparison_acc_f1.png"
)

total_end = time.time()
print("\n==============================================")
print(f"Total elapsed time: {total_end - total_start:.1f} seconds")
print("Done.")
