# models/rf_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def tune_rf(X_train, y_train, X_val, y_val):
    """
    Performs hyperparameter tuning for a Random Forest classifier using a
    simple grid search strategy.

    The model is trained on (X_train, y_train) and evaluated on
    (X_val, y_val). The selection criterion is the macro F1-score.

    Parameters:
    - X_train, y_train: training feature vectors and labels
    - X_val, y_val    : validation feature vectors and labels

    Returns:
    - best_model : trained Random Forest model with the best parameters
    - best_params: tuple (n_estimators, max_depth, min_samples_split)
    """

    # Number of trees in the forest.
    # Values were selected based on previous exploratory experiments.
    n_estimators = [100, 300, 500]

    # Maximum depth of each tree.
    # None allows fully grown trees; 20 limits depth to reduce overfitting.
    max_depths = [None, 20]

    # Minimum number of samples required to split an internal node.
    min_splits = [2, 5]

    # Keep track of the best performing configuration
    best_f1 = -1.0
    best_model = None
    best_params = None

    # Grid search over all parameter combinations
    for n in n_estimators:
        for d in max_depths:
            for m in min_splits:

                # Initialize Random Forest with current parameters
                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=d,
                    min_samples_split=m,
                    n_jobs=-1,        # Use all available CPU cores
                    random_state=42   # Ensure reproducibility
                )

                # Train the model on the training set
                model.fit(X_train, y_train)

                # Predict labels for the validation set
                preds = model.predict(X_val)

                # Compute macro F1-score (balanced across classes)
                f1 = precision_recall_fscore_support(
                    y_val,
                    preds,
                    average="macro",
                    zero_division=0
                )[2]

                # Update the best model if performance improves
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_params = (n, d, m)

    return best_model, best_params
