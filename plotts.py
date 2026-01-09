import importingfile as i

def plot_residual(residuals, y_test, name):
    avg_residual = i.np.average(residuals)
    print(avg_residual)

    # Scatter plot
    i.plt.scatter(y_test, residuals, alpha=0.5, label=name)
    i.plt.axhline(0, linestyle="--")

    i.plt.xlabel("Predicted quality")
    i.plt.ylabel("Residual")
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
