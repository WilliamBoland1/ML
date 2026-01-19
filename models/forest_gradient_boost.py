from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.inspection import permutation_importance

from catboost import CatBoostRegressor
from scipy.stats import randint, uniform


def catboost_regression(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    do_plot: bool = True,
) -> Dict[str, Any]:

    X = df.drop(columns=["quality"])
    y = df["quality"].astype(float)

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: create validation set from training data
    # Adjusted val_size to account for the first split
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, random_state=random_state
    )

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    param_dist = {
        "iterations": randint(400, 2000),
        "depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.2),
        "l2_leaf_reg": uniform(1.0, 8.0),
        "bagging_temperature": uniform(0.0, 1.0),
        "random_strength": uniform(0.0, 2.0),
        "min_data_in_leaf": randint(1, 40),
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=60,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )

    # Use validation set for early stopping instead of test set
    search.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=80,
    )

    best = search.best_estimator_
    y_pred = best.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)  # no deprecation warning
    r2 = r2_score(y_test, y_pred)

    perm = permutation_importance(
        best, X_test, y_test,
        n_repeats=25,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    if do_plot:
        _plot_pred_vs_true(y_test, y_pred)
        _plot_residuals(y_test - y_pred)
        _plot_catboost_feature_importance(best, X.columns)
        _plot_permutation_importance(perm.importances_mean, X.columns)

    return {
        "model": best,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
        "y_test": y_test,
        "permutation_importances": perm.importances_mean,
        "best_params": search.best_params_,
    }


# =========================
# Plotting helpers
# =========================

def _plot_pred_vs_true(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("True quality")
    plt.ylabel("Predicted quality")
    plt.title("Predicted vs True (CatBoost)")
    plt.tight_layout()
    plt.show()


def _plot_residuals(residuals):
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual (true - pred)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()


def _plot_catboost_feature_importance(model: CatBoostRegressor, feature_names):
    importances = model.get_feature_importance()
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=importances[idx],
        y=[feature_names[i] for i in idx]
    )
    plt.title("CatBoost Feature Importances")
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
    plt.xlabel("Mean importance (RMSE impact)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# =========================
# Run
# =========================

if __name__ == "__main__":
    df = pd.read_csv("data/winequality-red.csv", sep=";")
    res = catboost_regression(df, do_plot=True)

    print(f"MAE:  {res['mae']:.4f}")
    print(f"RMSE: {res['rmse']:.4f}")
    print(f"R²:   {res['r2']:.4f}")
    print("Best params:", res["best_params"])
