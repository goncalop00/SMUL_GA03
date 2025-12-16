# evaluation.py
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate(model, X_test, y_test):
    t0 = time.time()
    preds = model.predict(X_test)
    pred_time = time.time() - t0

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    return acc, prec, rec, f1, preds, pred_time
