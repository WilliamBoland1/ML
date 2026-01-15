import utils.importingfile as i


def _to_numpy(x):
    """Convert pandas Series/Index or list-like to numpy array safely."""
    return x.to_numpy() if hasattr(x, "to_numpy") else i.np.array(x)


def plot_residual_hist(results, bins=30, title="Residual Distribution"):
    """
    results: list of tuples (name, y_test, y_pred)
    """
    i.plt.figure(figsize=(10, 6))

    for name, y_test, y_pred in results:
        y_t = _to_numpy(y_test)
        y_p = _to_numpy(y_pred)
        residuals = y_t - y_p
        i.plt.hist(residuals, bins=bins, alpha=0.4, label=name)

    i.plt.axvline(0, linestyle="--")
    i.plt.xlabel("Residual (true - predicted)")
    i.plt.ylabel("Count")
    i.plt.title(title)
    i.plt.legend()
    i.plt.tight_layout()


def plot_mae_by_quality(results, title="MAE by True Quality Level"):
    """
    Shows where models perform worse (typically extremes like 3 or 8).
    results: list of tuples (name, y_test, y_pred)
    """
    i.plt.figure(figsize=(10, 6))

    for name, y_test, y_pred in results:
        y_t = _to_numpy(y_test)
        y_p = _to_numpy(y_pred)

        levels = sorted(set(y_t))
        maes = []
        for q in levels:
            mask = (y_t == q)
            maes.append(i.np.mean(i.np.abs(y_t[mask] - y_p[mask])))

        i.plt.plot(levels, maes, marker="o", label=name)

    i.plt.xlabel("True quality level")
    i.plt.ylabel("MAE")
    i.plt.title(title)
    i.plt.legend()
    i.plt.tight_layout()


def plot_top_coefficients(model, feature_names, title="Top coefficients", top_n=10):
    """
    model must have .coef_ attribute (e.g., Ridge, Lasso, LinearRegression).
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not have coef_. Pass a linear model like Ridge/Lasso.")

    coefs = _to_numpy(model.coef_)

    # Sort by absolute magnitude
    idx = i.np.argsort(i.np.abs(coefs))[::-1][:top_n]
    top_features = [feature_names[j] for j in idx]
    top_values = coefs[idx]

    i.plt.figure(figsize=(10, 6))
    i.plt.barh(top_features[::-1], top_values[::-1])
    i.plt.xlabel("Coefficient value")
    i.plt.title(title)
    i.plt.tight_layout()


def plot_pred_vs_true(results, title="Predicted vs True"):
    """
    results: list of tuples (name, y_test, y_pred)
    """
    i.plt.figure(figsize=(10, 6))

    y_all = i.np.concatenate([_to_numpy(r[1]) for r in results])
    y_min, y_max = y_all.min(), y_all.max()

    

    for name, y_test, y_pred in results:
        plot_list = list(range(int(y_min), int(y_max)+1))
        y_t = _to_numpy(y_test)
        y_p = _to_numpy(y_pred)
        i.plt.scatter(y_t, y_p, alpha=0.35, label=name)

        # Calculate average predicted value for each true quality level
        avg_list_sum_list = [0] * 10
        avg_count_list = [0] * 10
        
        for one in range(len(y_t)):
            index = y_t[one]-1
            avg_list_sum_list[index] += y_p[one]
            avg_count_list[index] += 1
        
        
        avg_list = [avg_list_sum_list[i] / avg_count_list[i] if avg_count_list[i] > 0 else 0 for i in range(10)]
        
        avg_list = [v for v in avg_list if v != 0]


        i.plt.plot(plot_list, avg_list, marker="o", linestyle="-", label=f"{name} Avg per level")
    
    
    # Perfect prediction line
    i.plt.plot([y_min, y_max], [y_min, y_max], linestyle="--", label="Perfect")
    
    i.plt.grid(True)
    
    i.plt.xlabel("True quality")
    i.plt.ylabel("Predicted quality")
    i.plt.title(title)
    i.plt.legend()
    i.plt.tight_layout()


def plot_residual(residuals, y_pred, name):
    """
    Residual plot: x = predictions, y = residuals (true - predicted)
    Multiple calls before plt.show() will overlay multiple models.
    """
    residuals = _to_numpy(residuals)
    y_pred = _to_numpy(y_pred)

    avg_residual = i.np.average(residuals)
    print(f"{name} avg residual: {avg_residual:.6f}")

    # Scatter plot
    i.plt.scatter(y_pred, residuals, alpha=0.5, label=name)
    i.plt.axhline(0, linestyle="--")

    i.plt.xlabel("Predicted quality")
    i.plt.ylabel("Residual (true - predicted)")
    i.plt.title("Residuals vs Predictions")
    i.plt.legend()

    # Count existing text boxes and offset vertically
    ax = i.plt.gca()
    n_texts = len(ax.texts)

    i.plt.text(
        0.05, 0.95 - 0.07 * n_texts,
        f"{name} avg residual: {avg_residual:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.8)
    )
