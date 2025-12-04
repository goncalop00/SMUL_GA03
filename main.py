import soundata
import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from tqdm import tqdm

# -----------------------------------------------------
# 1. Load Dataset using soundata
# -----------------------------------------------------
print("Loading UrbanSound8K...")
dataset = soundata.initialize('urbansound8k')
dataset.download()
dataset.validate()
import pandas as pd

CLASS_MAP = {
    "air_conditioner": 0,
    "car_horn": 1,
    "children_playing": 2,
    "dog_bark": 3,
    "drilling": 4,
    "engine_idling": 5,
    "gun_shot": 6,
    "jackhammer": 7,
    "siren": 8,
    "street_music": 9
}

# Build minimal metadata table
rows = []

for clip_id in dataset.clip_ids:
    clip = dataset.clip(clip_id)

    filename = clip.audio_path.split("/")[-1]
    class_id = int(filename.split("-")[1])   # second field in filename

    rows.append({
        "audio_path": clip.audio_path,
        "fold": clip.fold,
        "classID": class_id
    })

meta = pd.DataFrame(rows)



# -----------------------------------------------------
# 3. Utility function to load audio
# -----------------------------------------------------
def load_audio(audio_path, sr=None):
    audio, sr = librosa.load(audio_path, sr=sr)  # sr=None keeps original rate
    return audio, sr


# -----------------------------------------------------
# 4. Paper-based MFCC Feature Extraction (225 dims)
# -----------------------------------------------------
def extract_mfcc(audio, sr):
    frame_length = int(0.0232 * sr)
    hop_length = frame_length // 2

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=25,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=40
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 5 summary stats for MFCCs
    def stats_5(matrix):
        return np.concatenate([
            np.min(matrix, axis=1),
            np.max(matrix, axis=1),
            np.median(matrix, axis=1),
            np.mean(matrix, axis=1),
            np.var(matrix, axis=1)
        ])

    mfcc_stats = stats_5(mfcc)

    delta_mean = np.mean(delta, axis=1)
    delta_var = np.var(delta, axis=1)

    delta2_mean = np.mean(delta2, axis=1)
    delta2_var = np.var(delta2, axis=1)

    return np.concatenate([
        mfcc_stats,  # 125 dims
        delta_mean,  # 25
        delta_var,   # 25
        delta2_mean, # 25
        delta2_var   # 25
    ])  # total = 225 dims


# -----------------------------------------------------
# 5. Build dataset for a given test fold
# -----------------------------------------------------
def build_fold_dataset(test_fold):
    print(f"\nBuilding dataset for fold {test_fold}...")

    train_rows = meta[meta["fold"] != test_fold]
    test_rows  = meta[meta["fold"] == test_fold]

    X_train, y_train = [], []
    X_test,  y_test  = [], []

    # Training set
    for _, row in tqdm(train_rows.iterrows(), total=len(train_rows), desc="Training data"):
        audio, sr = load_audio(row.audio_path, sr=None)
        feat = extract_mfcc(audio, sr)
        X_train.append(feat)
        y_train.append(row.classID)

    # Test set
    for _, row in tqdm(test_rows.iterrows(), total=len(test_rows), desc="Test data"):
        audio, sr = load_audio(row.audio_path, sr=None)
        feat = extract_mfcc(audio, sr)
        X_test.append(feat)
        y_test.append(row.classID)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test),  np.array(y_test)
    )


# -----------------------------------------------------
# 6. Train SVM Model
# -----------------------------------------------------
def train_svm(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='scale')
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------
# 7. Evaluate Model
# -----------------------------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='macro', zero_division=0
    )
    return acc, precision, recall, f1


# -----------------------------------------------------
# 8. FULL 10-FOLD EVALUATION
# -----------------------------------------------------
results = []
start_time = time.time()

for fold in range(1, 11):
    print("\n==============================================")
    print(f"               TEST FOLD {fold}")
    print("==============================================")

    X_train, y_train, X_test, y_test = build_fold_dataset(fold)
    model = train_svm(X_train, y_train)
    acc, prec, rec, f1 = evaluate(model, X_test, y_test)

    print(f"Fold {fold} → Accuracy: {acc:.3f}, F1: {f1:.3f}")
    results.append([acc, prec, rec, f1])

results = np.array(results)
end_time = time.time()


# -----------------------------------------------------
# 9. Print Final Mean Results
# -----------------------------------------------------
mean_acc, mean_prec, mean_rec, mean_f1 = np.mean(results, axis=0)

print("\n================ FINAL RESULTS ================")
print(f"Mean Accuracy : {mean_acc:.3f}")
print(f"Mean Precision: {mean_prec:.3f}")
print(f"Mean Recall   : {mean_rec:.3f}")
print(f"Mean F1-score : {mean_f1:.3f}")
print("===============================================")
print(f"Total time: {end_time - start_time:.1f} seconds")
