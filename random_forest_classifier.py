from typing import Dict, Any, Optional
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance

def random_forest_classifier(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, do_plot: bool = True) -> Dict[str, Any]:
    X = df.drop("quality", axis=1)
    y = df["quality"].astype(int)  # ensure integer class labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=random_state))
    ])

    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # permutation importance (on the pipeline; pipeline includes scaler)
    r = permutation_importance(
        best,
        X_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring="accuracy",
        n_jobs=-1
    )
    perm_importances = r.importances_mean

    if do_plot:
        _plot_confusion_matrix(cm, sorted(y.unique()))
        _plot_feature_importances(best, X.columns)
        _plot_permutation_importance(perm_importances, X.columns)

    return {"model": best, "y_test": y_test, "y_pred": y_pred, "accuracy": acc, "report": report, "confusion_matrix": cm, "permutation_importances": perm_importances}

def _plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def _plot_feature_importances(pipe: Pipeline, feature_names):
    # extract the fitted RandomForestClassifier from the pipeline
    rf = pipe.named_steps.get("rf")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return
    importances = rf.feature_importances_
    idx = importances.argsort()[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx], y=[feature_names[i] for i in idx])
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def _plot_permutation_importance(importances, feature_names):
    idx = importances.argsort()[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx], y=[feature_names[i] for i in idx])
    plt.title("Permutation Importances (mean over repeats)")
    plt.xlabel("Importance (mean)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/winequality-red.csv", sep=";")

    res = random_forest_classifier(df)

    print(f"Accuracy: {res['accuracy']:.4f}")
    print(res["report"])


