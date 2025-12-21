# config.py

RUN_SINGLE_FOLD = False
SINGLE_FOLD = 1
TUNE = True
DATA_AUGMENTED = False


CONFIG = {
    "EXECUTION": {
        "RUN_SINGLE_FOLD": RUN_SINGLE_FOLD,
        "SINGLE_FOLD": SINGLE_FOLD if RUN_SINGLE_FOLD else "ALL (1–10)",
        "TUNE": TUNE,
    },
    "FEATURES": {
        "type": "MFCC",
        "n_mfcc": 40,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 90,
        "fmax_ratio": 0.5,
        "stats": ["mean", "std"],
        "delta": True,
        "delta_delta": True,
    },
    "DATASET": {
        "name": "UrbanSound8K",
        "num_classes": 10,
        "evaluation": "Official 10-fold cross-validation",
    },
    "METRICS": {
        "primary": "macro-F1",
        "secondary": ["accuracy", "precision", "recall"],
    },
}

def print_experiment_config(config):
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION ")
    print("=" * 70)

    for section, values in config.items():
        print(f"\n[{section}]")
        for k, v in values.items():
            print(f"{k:25}: {v}")

    print("=" * 70 + "\n")
