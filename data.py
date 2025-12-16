# data.py
import numpy as np
import soundata
import librosa
import pandas as pd
from tqdm import tqdm


def load_dataset():
    dataset = soundata.initialize("urbansound8k")
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

def build_fold_dataset(meta, extract_fn, test_fold):
    train = meta[meta.fold != test_fold]
    test  = meta[meta.fold == test_fold]

    X_train, y_train, X_test, y_test = [], [], [], []

    for _, r in tqdm(train.iterrows(), total=len(train), desc="Training data"):
        audio, sr = load_audio(r.audio_path)
        X_train.append(extract_fn(audio, sr))
        y_train.append(r.classID)

    for _, r in tqdm(test.iterrows(), total=len(test), desc="Test data"):
        audio, sr = load_audio(r.audio_path)
        X_test.append(extract_fn(audio, sr))
        y_test.append(r.classID)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test),  np.array(y_test)
    )
