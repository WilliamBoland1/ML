"""
Evaluation module for model performance metrics
"""

import utils.importingfile as i


def evaluate_model(y_test, y_pred, model_name):
    """
    Evaluate a model using various metrics.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    mse = i.mean_squared_error(y_test, y_pred)
    rmse = i.root_mean_squared_error(y_test, y_pred)
    mae = i.mean_absolute_error(y_test, y_pred)
    r2 = i.r2_score(y_test, y_pred)
    
    metrics = {
        "model": model_name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R²:   {r2:.4f}")
    
    return metrics


def evaluate_all_models(models_dict):
    """
    Evaluate multiple models and return all metrics.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with format:
        {
            "model_name": (y_test, y_pred),
            ...
        }
    
    Returns:
    --------
    list : List of dictionaries containing metrics for each model
    """
    all_metrics = []
    
    for model_name, (y_test, y_pred) in models_dict.items():
        metrics = evaluate_model(y_test, y_pred, model_name)
        all_metrics.append(metrics)
    
    return all_metrics
