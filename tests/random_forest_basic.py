from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.inspection import permutation_importance


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


def random_forest_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    do_plot: bool = True
) -> Dict[str, Any]:
    """
    Train and evaluate a Random Forest classifier on a wine quality dataset.
    """

    # Separate features and target variable
    X = df.drop("quality", axis=1)
    y = df["quality"].astype(int)

    # Split data into train and test sets with class stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Initialize Random Forest with balanced class weights
    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced"
    )

    # Stratified k-fold cross-validation
    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=random_state
    )

    # Hyperparameter distributions for randomized search
    param_dist = {
        "n_estimators": randint(300, 900),
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", 0.5],
        "bootstrap": [True, False],
    }

    # Randomized hyperparameter search
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=60,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=random_state
    )

    # Train models and select best estimator
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Predict on test set
    y_pred = best.predict(X_test)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Permutation feature importance
    r = permutation_importance(
        best,
        X_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

    # # Plot results if enabled
    # if do_plot:
    #     _plot_confusion_matrix(cm, sorted(y.unique()))
    #     _plot_feature_importances(best, X.columns)
    #     _plot_permutation_importance(r.importances_mean, X.columns)

    # Return all relevant outputs
    return {
        "model": best,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "report": report,
        "confusion_matrix": cm,
        "permutation_importances": r.importances_mean,
    }



# =========================
# Plotting helpers
# =========================

def _plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted quality")
    plt.ylabel("True quality")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def _plot_feature_importances(model: RandomForestClassifier, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[i] for i in idx]
    )
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def _plot_permutation_importance(importances, feature_names):
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[i] for i in idx]
    )
    plt.title("Permutation Importances (mean over repeats)")
    plt.xlabel("Mean importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# =========================
# Run
# =========================

if __name__ == "__main__":
    df = pd.read_csv("data/winequality-red.csv", sep=";")

    res = random_forest_classifier(df)

    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Balanced accuracy: {res['balanced_accuracy']:.4f}")
    print(res["report"])

# if __name__ == "__main__":
#     # Load data
#     df = pd.read_csv("data/winequality-red.csv", sep=";")

#     seeds = range(10)  # 0–9
#     results = []

#     for seed in seeds:
#         print(f"Running with seed {seed}...")
#         res = random_forest_classifier(
#             df,
#             random_state=seed,
#             do_plot=False  # disable plots during benchmarking
#         )

#         results.append({
#             "seed": seed,
#             "accuracy": res["accuracy"],
#             "balanced_accuracy": res["balanced_accuracy"],
#         })

#     # Convert to DataFrame for analysis
#     results_df = pd.DataFrame(results)

#     print(results_df)
#     print("\nSummary statistics:")
#     print(results_df[["accuracy", "balanced_accuracy"]].describe())