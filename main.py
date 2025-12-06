import soundata
import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
# CONFIGURAÇÃO DE EXECUÇÃO
# -----------------------------------------------------
RUN_SINGLE_FOLD = False   # True para correr só um fold (debug)
SINGLE_FOLD = 1           # valor entre 1 e 10 quando RUN_SINGLE_FOLD = True

# -----------------------------------------------------
# 1. Load Dataset using soundata
# -----------------------------------------------------
print("Loading UrbanSound8K...")
dataset = soundata.initialize('urbansound8k')
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
        SVC(kernel='rbf', C=10, gamma='scale', probability=False)
    )
    model.fit(X_train, y_train)
    return model

# -----------------------------------------------------
# 6b. Train Random Forest (nova função)
# -----------------------------------------------------
def train_random_forest(X_train, y_train, n_estimators=100):
    # RandomForest não necessita de StandardScaler; mantemos features originais
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# -----------------------------------------------------
# 7. Evaluate Model (agora também devolve preds)
# -----------------------------------------------------
def evaluate(model, X_test, y_test):
    start_pred = time.time()
    preds = model.predict(X_test)
    end_pred = time.time()
    pred_time = end_pred - start_pred

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='macro', zero_division=0
    )
    return acc, precision, recall, f1, preds, pred_time

# -----------------------------------------------------
# 8. FULL 10-FOLD EVALUATION (AGORA COM RF)
# -----------------------------------------------------
if RUN_SINGLE_FOLD:
    folds_to_run = [SINGLE_FOLD]
else:
    folds_to_run = range(1, 11)

# Definir modelos a testar
model_constructors = {
    "SVM": train_svm,
    "RandomForest": train_random_forest
}

# Estruturas para guardar resultados por modelo
all_results = {}
total_start = time.time()

for model_name, train_fn in model_constructors.items():
    print("\n==============================================")
    print(f"        MODEL: {model_name}")
    print("==============================================")
    fold_metrics = []   # [fold, acc, prec, rec, f1, train_time, pred_time]
    conf_matrices = []
    for fold in folds_to_run:
        print("\n----------------------------------------------")
        print(f"               TEST FOLD {fold} - {model_name}")
        print("----------------------------------------------")

        X_train, y_train, X_test, y_test = build_fold_dataset(fold)

        # Treino e temporização
        t0 = time.time()
        if model_name == "SVM":
            model = train_fn(X_train, y_train)  # train_svm
        else:
            # Para RF, podes ajustar n_estimators se quiseres
            model = train_fn(X_train, y_train, n_estimators=100)
        t1 = time.time()
        train_time = t1 - t0

        acc, prec, rec, f1, preds, pred_time = evaluate(model, X_test, y_test)

        print(f"{model_name} Fold {fold} → Accuracy: {acc:.3f}, F1: {f1:.3f}, train_time: {train_time:.1f}s, pred_time: {pred_time:.2f}s")

        fold_metrics.append([fold, acc, prec, rec, f1, train_time, pred_time])
        cm = confusion_matrix(y_test, preds, labels=list(range(10)))
        conf_matrices.append(cm)

    # guardar resultados deste modelo
    df_metrics = pd.DataFrame(
        fold_metrics,
        columns=["Fold", "Accuracy", "Precision", "Recall", "F1", "TrainTime_s", "PredTime_s"]
    )

    mean_acc  = df_metrics["Accuracy"].mean()
    mean_prec = df_metrics["Precision"].mean()
    mean_rec  = df_metrics["Recall"].mean()
    mean_f1   = df_metrics["F1"].mean()
    mean_train_time = df_metrics["TrainTime_s"].mean()
    mean_pred_time  = df_metrics["PredTime_s"].mean()
    mean_cm = np.mean(conf_matrices, axis=0)

    all_results[model_name] = {
        "df_metrics": df_metrics,
        "mean_acc": mean_acc,
        "mean_prec": mean_prec,
        "mean_rec": mean_rec,
        "mean_f1": mean_f1,
        "mean_train_time": mean_train_time,
        "mean_pred_time": mean_pred_time,
        "mean_confusion": mean_cm
    }

    # salvar resultados do modelo
    df_metrics.to_csv(f"fold_metrics_{model_name}.csv", index=False)
    np.savetxt(f"mean_confusion_matrix_{model_name}.csv", mean_cm, delimiter=",")
    print(f"\nSaved metrics for {model_name} to 'fold_metrics_{model_name}.csv' and 'mean_confusion_matrix_{model_name}.csv'.")

total_end = time.time()
print(f"\nTotal elapsed time for all experiments: {total_end - total_start:.1f} seconds")

# -----------------------------------------------------
# 9. Mostrar comparação resumida
# -----------------------------------------------------
print("\n================ COMPARISON SUMMARY ================")
summary_rows = []
for name, res in all_results.items():
    print(f"\nModel: {name}")
    print(f" Mean Accuracy : {res['mean_acc']:.3f}")
    print(f" Mean Precision: {res['mean_prec']:.3f}")
    print(f" Mean Recall   : {res['mean_rec']:.3f}")
    print(f" Mean F1-score : {res['mean_f1']:.3f}")
    print(f" Mean Train Time (s): {res['mean_train_time']:.2f}")
    print(f" Mean Pred Time  (s): {res['mean_pred_time']:.4f}")

    summary_rows.append({
        "Model": name,
        "MeanAcc": res['mean_acc'],
        "MeanF1": res['mean_f1'],
        "MeanTrain_s": res['mean_train_time'],
        "MeanPred_s": res['mean_pred_time']
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("comparison_summary.csv", index=False)
print("\nSaved comparison summary to 'comparison_summary.csv'.")

# -----------------------------------------------------
# 10. Plots de comparação (Acc & F1)
# -----------------------------------------------------
plt.figure(figsize=(6,4))
x = np.arange(len(df_summary))
plt.bar(x - 0.15, df_summary["MeanAcc"], width=0.3, label="Mean Accuracy")
plt.bar(x + 0.15, df_summary["MeanF1"], width=0.3, label="Mean F1")
plt.xticks(x, df_summary["Model"])
plt.ylabel("Score")
plt.title("Comparison: Mean Accuracy and Mean F1")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_acc_f1.png")
plt.show()

# -----------------------------------------------------
# 11. Plotagem das matrizes de confusão médias (uma por modelo)
# -----------------------------------------------------
for name, res in all_results.items():
    mean_cm = res["mean_confusion"]
    plt.figure(figsize=(8,6))
    plt.imshow(mean_cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Mean Confusion Matrix ({name})")
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
    plt.savefig(f"mean_confusion_matrix_{name}.png")
    plt.show()

print("\nDone.")
