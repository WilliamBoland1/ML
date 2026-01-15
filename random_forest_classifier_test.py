from typing import Dict, Any
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.inspection import permutation_importance

def _group_quality_series(series: pd.Series) -> (pd.Series, list):
    # Default grouping: low (<=4), medium (5-6), high (>=7)
    def map_q(q):
        if q <= 4:
            return 0
        if q <= 6:
            return 1
        return 2
    mapped = series.apply(map_q).astype(int)
    labels = ["low", "medium", "high"]
    return mapped, labels

def _binary_quality_series(series: pd.Series, cutoff: float = 6.5) -> (pd.Series, list):
    # good if quality > cutoff (default cutoff 6.5 => quality >=7 is "good")
    mapped = (series > cutoff).astype(int)
    labels = ["not_good", "good"]
    return mapped, labels

def random_forest_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    do_plot: bool = True,
    group_quality: bool = False,
    use_class_weight: bool = False,
    evaluate_ordinal: bool = False,
    binary_quality: bool = False,
    binary_cutoff: float = 6.5
) -> Dict[str, Any]:
    X = df.drop("quality", axis=1)
    y_raw = df["quality"]

    if binary_quality:
        y, label_names = _binary_quality_series(y_raw, cutoff=binary_cutoff)
    elif group_quality:
        y, label_names = _group_quality_series(y_raw)
    else:
        y = y_raw.astype(int)
        label_names = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf_kwargs = {"random_state": random_state}
    # use simple, safe class-weight handling
    if use_class_weight:
        clf_kwargs["class_weight"] = "balanced"
    # build a weight grid depending on grouping (still used for optional GridSearch)
    if group_quality:
        weight_grid = [
            None,
            "balanced",
            {0: 1, 1: 1, 2: 1},
            {0: 2, 1: 1, 2: 2},
            {0: 3, 1: 1, 2: 3},
            {0: 3, 1: 1, 2: 5},
            {0: 4, 1: 1, 2: 6},
        ]
    else:
        classes = sorted(y.unique())
        # build weight variants programmatically so keys always match `classes`
        # examples: None, "balanced", symmetric/emphasis-on-extremes, emphasis-on-high, emphasis-on-low
        weight_grid = [None, "balanced"]

        # simple patterns: uniform, emphasize first, emphasize last, linear scale, inverse linear
        uniform = {c: 1 for c in classes}
        emphasize_first = {c: (len(classes) - i) for i, c in enumerate(classes)}   # higher weight to low-end
        emphasize_last = {c: (i + 1) for i, c in enumerate(classes)}              # higher weight to high-end
        extremes = {c: (1 if i not in (0, len(classes)-1) else 4) for i, c in enumerate(classes)}
        linear = {c: (i + 1) for i, c in enumerate(classes)}
        inverse_linear = {c: (len(classes) - i) for i, c in enumerate(classes)}

        weight_grid += [uniform, emphasize_first, emphasize_last, extremes, linear, inverse_linear]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(**clf_kwargs))
    ])

    param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    }

    if use_class_weight:
        param_grid["rf__class_weight"] = weight_grid


    # grid = GridSearchCV(pipe, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
    grid = GridSearchCV(pipe, param_grid, cv=4, scoring="balanced_accuracy", n_jobs=-1)

    grid.fit(X_train, y_train)

    # print best params and CV score
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)


    bal_acc = balanced_accuracy_score(y_test, y_pred)  # <-- add
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # permutation importance (pipeline handles scaling)
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

    # Ordinal / ordered evaluation (useful when labels have order)
    ordinal_mae = None
    kappa_qwk = None
    if evaluate_ordinal:
        # If grouped, labels are already integers 0..2; otherwise raw quality integers are used
        ordinal_mae = mean_absolute_error(y_test, y_pred)
        # quadratic weighted kappa (treats labels as ordered)
        try:
            kappa_qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        except Exception:
            kappa_qwk = None

    if do_plot:
        _plot_confusion_matrix(cm, label_names)
        _plot_feature_importances(best, X.columns)
        _plot_permutation_importance(perm_importances, X.columns)

    return {
        "model": best,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "report": report,
        "confusion_matrix": cm,
        "permutation_importances": perm_importances,
        "ordinal_mae": ordinal_mae,
        "kappa_quadratic": kappa_qwk
    }

def _plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def _plot_feature_importances(pipe: Pipeline, feature_names):
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
    parser = argparse.ArgumentParser(description="Train and evaluate Random Forest classifier on a dataset with a 'quality' column.")
    parser.add_argument("--data", "-d", required=True, help="Path to CSV file")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--group", action="store_true", help="Group quality into categories (low/medium/high)")
    parser.add_argument("--class-weight", action="store_true", help="Use balanced class weights")
    parser.add_argument("--ordinal", action="store_true", help="Compute ordinal metrics (MAE, quadratic kappa)")
    parser.add_argument("--binary", action="store_true", help="Binary target: good (quality >= 7) vs not good")
    parser.add_argument("--cutoff", type=float, default=6.5, help="Cutoff for binary target (default 6.5 -> good means >= 7)")
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep=";")
    res = random_forest_classifier(
        df,
        do_plot=not args.no_plot,
        group_quality=args.group,
        use_class_weight=args.class_weight,
        evaluate_ordinal=args.ordinal,
        binary_quality=args.binary,
        binary_cutoff=args.cutoff
    )

    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Balanced accuracy: {res['balanced_accuracy']:.4f}")
    print(res["report"])
    if args.ordinal:
        print(f"Ordinal MAE: {res['ordinal_mae']}")
        print(f"Quadratic weighted kappa: {res['kappa_quadratic']}")


