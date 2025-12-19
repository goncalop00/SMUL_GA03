# svm_model.py
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support


def tune_svm(X_train, y_train, X_val, y_val):
    """
    Performs hyperparameter tuning for a Support Vector Machine (SVM)
    classifier with an RBF kernel using a reduced grid search.

    The model is trained on (X_train, y_train) and evaluated on
    (X_val, y_val). The best configuration is selected based on
    the macro F1-score.

    Parameters:
    - X_train, y_train: training feature vectors and labels
    - X_val, y_val    : validation feature vectors and labels

    Returns:
    - best_model : trained SVM model with the best hyperparameters
    - best_params: tuple (C, gamma)
    """

    # Regularization parameter.
    # Values were selected based on prior experiments showing convergence.
    C_values = [1, 5]

    # Kernel coefficient for the RBF kernel.
    # 'scale' is the default and usually robust; 'auto' is tested for comparison.
    gamma_values = ["scale", "auto"]

    # Track the best-performing configuration
    best_f1 = -1.0
    best_model = None
    best_params = None

    # Grid search over hyperparameter combinations
    for C in C_values:
        for g in gamma_values:

            # Build a pipeline:
            # - StandardScaler normalizes features
            # - SVC performs classification with RBF kernel
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=C, gamma=g)
            )

            # Train the model on the training set
            model.fit(X_train, y_train)

            # Predict labels for the validation set
            preds = model.predict(X_val)

            # Compute macro F1-score (balanced across all classes)
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
                best_params = (C, g)

    return best_model, best_params
