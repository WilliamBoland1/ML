import ridge_regression as rr
import Polinomial_regression_MSE as prm
import Pol_pluss_ridge as ppr
import lasso as l
import plotts as p
import random_forest_basic as rfb
import random_forest_regression as rfr
import utils.evaluation as ev
import forest_gradient_boost as fgb

import utils.importingfile as i
import pandas as pd
import numpy as np


def run_single_iteration(df, random_state):
    """
    Run all models once with a specific random_state and return their predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary with y_test and y_pred for each model
    """
    results = {}
    
    # Ridge (needs random_state parameter)
    y_test_rrr, y_pred_rrr = rr.ridge_regression(df, random_state=random_state)
    results["Ridge"] = (y_test_rrr, y_pred_rrr)

    # Polynomial regression (needs random_state parameter)
    X_test_poly_scaled, y_test_prm, y_pred_prm = prm.polynomial_regression(df, degree=2, random_state=random_state)
    results["Polynomial"] = (y_test_prm, y_pred_prm)

    # Polynomial + Ridge (needs random_state parameter)
    y_test_ppr, y_pred_ppr, best_params_ppr = ppr.poly_ridge_regression(df, random_state=random_state)
    results["Poly+Ridge"] = (y_test_ppr, y_pred_ppr)

    # Lasso (needs random_state parameter)
    y_test_lasso, y_pred_lasso, lasso_model = l.lasso_regression(df, random_state=random_state)
    results["Lasso"] = (y_test_lasso, y_pred_lasso)

    # Random Forest
    rf_results_regression = rfr.random_forest_regressor(df, random_state=random_state, do_plot=False)
    results["Random Forest"] = (rf_results_regression["y_test"], rf_results_regression["y_pred"])

    # Gradient Boosting with CatBoost (if it supports random_state)
    catboost_results = fgb.catboost_regression(df, random_state=random_state, do_plot=False)
    results["CatBoost"] = (catboost_results["y_test"], catboost_results["y_pred"])
    
    return results


def run_multiple_random_states(df, random_states, models_to_test=None):
    """
    Run all models multiple times with different random states and collect metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    random_states : list
        List of random states to test
    models_to_test : list, optional
        List of model names to test. If None, test all models.
    
    Returns:
    --------
    dict : Aggregated metrics with mean and std for each model
    """
    # Dictionary to store all metrics for each model
    model_metrics = {}
    
    print(f"Running models with {len(random_states)} different random states...")
    print(f"Random states: {random_states}\n")
    
    # Run each random state
    for iteration, rs in enumerate(random_states, 1):
        print(f"Running iteration {iteration}/{len(random_states)} (random_state={rs})...")
        
        # Get predictions for all models
        iteration_results = run_single_iteration(df, rs)
        
        # Evaluate each model
        models_to_evaluate = {name: preds for name, preds in iteration_results.items()}
        
        if models_to_test:
            models_to_evaluate = {k: v for k, v in models_to_evaluate.items() if k in models_to_test}
        
        # Get metrics without printing each iteration
        for model_name, (y_test, y_pred) in models_to_evaluate.items():
            metrics = {
                "mse": i.mean_squared_error(y_test, y_pred),
                "rmse": i.root_mean_squared_error(y_test, y_pred),
                "mae": i.mean_absolute_error(y_test, y_pred),
                "r2": i.r2_score(y_test, y_pred),
            }
            
            if model_name not in model_metrics:
                model_metrics[model_name] = {
                    "mse": [],
                    "rmse": [],
                    "mae": [],
                    "r2": [],
                }
            
            for metric_name, value in metrics.items():
                model_metrics[model_name][metric_name].append(value)
    
    # Calculate mean and std for each model and metric
    summary_stats = {}
    for model_name, metrics in model_metrics.items():
        summary_stats[model_name] = {}
        for metric_name, values in metrics.items():
            summary_stats[model_name][metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "all_values": values,
            }
    
    return summary_stats


def print_summary_comparison(summary_stats):
    """
    Print a nice formatted comparison of all models and their metrics.
    
    Parameters:
    -----------
    summary_stats : dict
        Dictionary with aggregated metrics for each model
    """
    print("\n" + "="*100)
    print("SUMMARY: Mean ± Std of Metrics Across All Random States")
    print("="*100)
    
    # Print detailed stats for each model
    for model_name in sorted(summary_stats.keys()):
        print(f"\n{model_name}:")
        print("-" * 50)
        for metric_name in ["mse", "rmse", "mae", "r2"]:
            stats = summary_stats[model_name][metric_name]
            print(f"  {metric_name.upper():5s}: {stats['mean']:7.4f} ± {stats['std']:7.4f} "
                  f"(min: {stats['min']:7.4f}, max: {stats['max']:7.4f})")
    
    # Print comparison table
    print("\n" + "="*100)
    print("COMPARISON TABLE (Mean Values)")
    print("="*100)
    
    metrics = ["mse", "rmse", "mae", "r2"]
    print(f"{'Model':<20}", end="")
    for metric in metrics:
        print(f"{metric.upper():>15}", end="")
    print()
    print("-" * 80)
    
    for model_name in sorted(summary_stats.keys()):
        print(f"{model_name:<20}", end="")
        for metric in metrics:
            mean_val = summary_stats[model_name][metric]["mean"]
            print(f"{mean_val:>15.4f}", end="")
        print()
    
    # Rank models by best mean score for each metric
    print("\n" + "="*100)
    print("MODEL RANKINGS (Best to Worst)")
    print("="*100)
    
    for metric in metrics:
        print(f"\n{metric.upper()} (lower is better for MSE/RMSE/MAE, higher is better for R²):")
        print("-" * 50)
        
        # Create ranking
        if metric == "r2":
            # For R², higher is better
            ranked = sorted(summary_stats.items(), 
                          key=lambda x: x[1][metric]["mean"], reverse=True)
        else:
            # For error metrics, lower is better
            ranked = sorted(summary_stats.items(), 
                          key=lambda x: x[1][metric]["mean"])
        
        for rank, (model_name, stats) in enumerate(ranked, 1):
            mean_val = stats[metric]["mean"]
            print(f"  {rank}. {model_name:<20} {mean_val:>10.4f}")


def main():
    # Load data
    df = i.pd.read_csv("data/winequality-red.csv", sep=";")  # or white
    
    # ----------------------- Configuration for Multiple Random States ----------------------
    
    # Define random states to test (different seeds for variability)
    random_states = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
    
    # Optional: test only specific models
    # models_to_test = ["Ridge", "Random Forest", "Lasso"]
    models_to_test = None  # Set to None to test all models
    
    # ----------------------- Run Multiple Iterations ----------------------
    
    summary_stats = run_multiple_random_states(df, random_states, models_to_test)
    
    # ----------------------- Display Results ----------------------
    
    print_summary_comparison(summary_stats)


if __name__ == "__main__":
    main()
