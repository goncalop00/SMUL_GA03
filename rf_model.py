# models/rf_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def tune_rf(X_train, y_train, X_val, y_val):
    n_estimators = [100, 300, 500]
    max_depths = [None, 20, 40]
    min_splits = [2, 5]

    best_f1 = -1
    best_model = None
    best_params = None

    for n in n_estimators:
        for d in max_depths:
            for m in min_splits:
                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=d,
                    min_samples_split=m,
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                f1 = precision_recall_fscore_support(
                    y_val, preds, average="macro", zero_division=0
                )[2]

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_params = (n, d, m)

    return best_model, best_params
