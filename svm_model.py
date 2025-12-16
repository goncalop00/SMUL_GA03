# models/svm_model.py
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

def tune_svm(X_train, y_train, X_val, y_val):
    C_values = [1, 5, 10, 20, 50]
    gamma_values = ["scale", "auto", 0.01, 0.1]

    best_f1 = -1
    best_model = None
    best_params = None

    for C in C_values:
        for g in gamma_values:
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=C, gamma=g)
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            f1 = precision_recall_fscore_support(
                y_val, preds, average="macro", zero_division=0
            )[2]

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_params = (C, g)

    return best_model, best_params
