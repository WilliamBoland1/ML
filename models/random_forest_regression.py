from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from scipy.stats import randint


def random_forest_regressor(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    do_plot: bool = True
) -> Dict[str, Any]:
    """
    Train and evaluate a Random Forest regressor on a wine quality dataset.
    """

    # Separate features and target
    X = df.drop("quality", axis=1)
    y = df["quality"].astype(float)  # keep numeric/continuous

    # Train/test split (no stratification for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1
    )

    # K-fold CV (not stratified)
    cv = KFold(
        n_splits=4,
        shuffle=True,
        random_state=random_state
    )

    # Hyperparameter distributions
    param_dist = {
        "n_estimators": randint(300, 900),
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", 0.5],
        "bootstrap": [True, False],
    }

    # Randomized search (optimize RMSE)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=60,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=random_state
    )

    # Train and select best model
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Predict
    y_pred = best.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Permutation importance (use RMSE; higher is worse, but importances still work fine)
    r = permutation_importance(
        best,
        X_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    if do_plot:
        _plot_pred_vs_true(y_test, y_pred)
        _plot_residuals(y_test - y_pred)
        _plot_feature_importances_reg(best, X.columns)
        _plot_permutation_importance_reg(r.importances_mean, X.columns)

    return {
        "model": best,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
        "y_test": y_test,
        "permutation_importances": r.importances_mean,
        "best_params": search.best_params_,
    }


# =========================
# Plotting helpers (regression)
# =========================

def _plot_pred_vs_true(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True quality")
    plt.ylabel("Predicted quality")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.show()


def _plot_residuals(residuals):
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual (true - pred)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()


def _plot_feature_importances_reg(model: RandomForestRegressor, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[i] for i in idx]
    )
    plt.title("Random Forest Feature Importances (Regressor)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def _plot_permutation_importance_reg(importances, feature_names):
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

    res = random_forest_regressor(df)

    print(f"MAE:  {res['mae']:.4f}")
    print(f"RMSE: {res['rmse']:.4f}")
    print(f"R²:   {res['r2']:.4f}")
    print("Best params:", res["best_params"])
