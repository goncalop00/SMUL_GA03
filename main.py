import soundata
import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Load Dataset using soundata
# -----------------------------------------------------
print("Loading UrbanSound8K...")
dataset = soundata.initialize('urbansound8k')
# Se já tens o dataset, podes comentar a próxima linha para não voltar a sacar:
# dataset.download()   # uncomment if you need to download the dataset 
dataset.validate()

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
# 4. MFCC Feature Extraction (fixed-size vector)
# -----------------------------------------------------
def extract_mfcc(audio, sr, n_mfcc=40):

    # MFCCs com parâmetros seguros (evita filtros vazios)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
        n_mels=40,
        fmax=sr * 0.45
    )

    # Garantir frames suficientes para delta
    if mfcc.shape[1] < 9:
        pad_amount = 9 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_amount)), mode='edge')

    delta = librosa.feature.delta(mfcc, width=3)
    delta2 = librosa.feature.delta(mfcc, order=2, width=3)

    feat = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        np.mean(delta2, axis=1),
        np.std(delta2, axis=1),
    ])

    return feat



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
# 7. Evaluate Model (agora também devolve preds)
# -----------------------------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='macro', zero_division=0
    )
    return acc, precision, recall, f1, preds


# -----------------------------------------------------
# 8. FULL 10-FOLD EVALUATION COM CONFUSION MATRICES
# -----------------------------------------------------
all_fold_metrics = []   # [fold, acc, prec, rec, f1]
conf_matrices = []      # uma confusion matrix por fold

start_time = time.time()

for fold in range(1, 11):
    print("\n==============================================")
    print(f"               TEST FOLD {fold}")
    print("==============================================")

    X_train, y_train, X_test, y_test = build_fold_dataset(fold)
    model = train_svm(X_train, y_train)
    acc, prec, rec, f1, preds = evaluate(model, X_test, y_test)

    print(f"Fold {fold} → Accuracy: {acc:.3f}, F1: {f1:.3f}")

    # guardar métricas deste fold
    all_fold_metrics.append([fold, acc, prec, rec, f1])

    # confusion matrix deste fold
    cm = confusion_matrix(y_test, preds, labels=list(range(10)))
    conf_matrices.append(cm)

end_time = time.time()


# -----------------------------------------------------
# 9. Tabela de métricas por fold + médias
# -----------------------------------------------------
df_metrics = pd.DataFrame(
    all_fold_metrics,
    columns=["Fold", "Accuracy", "Precision", "Recall", "F1"]
)

mean_acc  = df_metrics["Accuracy"].mean()
mean_prec = df_metrics["Precision"].mean()
mean_rec  = df_metrics["Recall"].mean()
mean_f1   = df_metrics["F1"].mean()

print("\n================ METRICS PER FOLD ================")
print(df_metrics)
print("==================================================")

print("\n================ FINAL MEAN RESULTS ================")
print(f"Mean Accuracy : {mean_acc:.3f}")
print(f"Mean Precision: {mean_prec:.3f}")
print(f"Mean Recall   : {mean_rec:.3f}")
print(f"Mean F1-score : {mean_f1:.3f}")
print("===================================================")
print(f"Total time: {end_time - start_time:.1f} seconds")


# -----------------------------------------------------
# 10. Mean Confusion Matrix
# -----------------------------------------------------
mean_cm = np.mean(conf_matrices, axis=0)

plt.figure(figsize=(8, 6))
plt.imshow(mean_cm, interpolation='nearest', cmap='Blues')
plt.title("Mean Confusion Matrix (10 folds)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.colorbar()

tick_marks = np.arange(10)
class_names = [
    "air_cond", "car_horn", "children", "dog_bark", "drilling",
    "engine", "gun_shot", "jackhammer", "siren", "street_music"
]
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 11. Guardar resultados em ficheiros (opcional mas útil p/ relatório)
# -----------------------------------------------------
df_metrics.to_csv("fold_metrics.csv", index=False)
np.savetxt("mean_confusion_matrix.csv", mean_cm, delimiter=",")
print("\nResults saved to 'fold_metrics.csv' and 'mean_confusion_matrix.csv'.")

