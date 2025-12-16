# features.py
import numpy as np
import librosa
from config import CONFIG

def extract_mfcc(audio, sr):
    cfg = CONFIG["FEATURES"]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=cfg["n_mfcc"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        fmax=sr * cfg["fmax_ratio"]
    )

    if mfcc.shape[1] < 9:
        mfcc = np.pad(mfcc, ((0,0),(0,9-mfcc.shape[1])), mode="edge")

    delta  = librosa.feature.delta(mfcc, width=3)
    delta2 = librosa.feature.delta(mfcc, order=2, width=3)

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ])
