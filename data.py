# data.py
import numpy as np
import soundata
import librosa
import pandas as pd
from tqdm import tqdm

from config import DATA_AUGMENTED


# -----------------------------------------------------
# Dataset utilities
# -----------------------------------------------------
def load_dataset():
    dataset = soundata.initialize("urbansound8k")
    dataset.download()
    dataset.validate()
    return dataset


def build_metadata(dataset):
    rows = []
    for clip_id in dataset.clip_ids:
        clip = dataset.clip(clip_id)
        filename = clip.audio_path.split("/")[-1]
        class_id = int(filename.split("-")[1])
        rows.append({
            "audio_path": clip.audio_path,
            "fold": clip.fold,
            "classID": class_id
        })
    return pd.DataFrame(rows)


def load_audio(audio_path, sr=None):
    audio, sr = librosa.load(audio_path, sr=sr)
    return audio, sr


# -----------------------------------------------------
# Data augmentation
# -----------------------------------------------------
# UrbanSound8K class IDs:
# 0: air_conditioner
# 4: drilling
# 5: engine_idling
# 6: gun_shot

AUGMENT_CLASSES = {
    0: (15, 30),   # air_conditioner
    4: (15, 30),   # drilling
    5: (15, 30),   # engine_idling
    6: (25, 35),   # gun_shot (lighter noise)
}


def add_noise(audio, snr_db):
    """Additive white Gaussian noise at a given SNR (dB)."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
    return audio + noise


# -----------------------------------------------------
# Fold construction
# -----------------------------------------------------
def build_fold_dataset(meta, extract_fn, test_fold):
    train = meta[meta.fold != test_fold]
    test  = meta[meta.fold == test_fold]

    X_train, y_train, X_test, y_test = [], [], [], []

    # -------------------------
    # Training data
    # -------------------------
    for _, r in tqdm(train.iterrows(), total=len(train), desc="Training data"):
        audio, sr = load_audio(r.audio_path)
        class_id = r.classID

        # Original sample
        X_train.append(extract_fn(audio, sr))
        y_train.append(class_id)

        # Selective additive noise augmentation
        if DATA_AUGMENTED and class_id in AUGMENT_CLASSES:
            snr_min, snr_max = AUGMENT_CLASSES[class_id]
            snr = np.random.uniform(snr_min, snr_max)

            audio_aug = add_noise(audio, snr)
            X_train.append(extract_fn(audio_aug, sr))
            y_train.append(class_id)

    # -------------------------
    # Test data (NO augmentation)
    # -------------------------
    for _, r in tqdm(test.iterrows(), total=len(test), desc="Test data"):
        audio, sr = load_audio(r.audio_path)
        X_test.append(extract_fn(audio, sr))
        y_test.append(r.classID)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test),  np.array(y_test)
    )
