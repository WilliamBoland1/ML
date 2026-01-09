import ridge_regression_red as rrr
import importingfile as i

df = i.pd.read_csv("data/winequality-red.csv", sep=";")  # or white

y_true = df["quality"]

y_test, y_rrr = rrr.ridge_regression(df)

residuals = y_test - y_rrr

i.plt.scatter(y_rrr, residuals, alpha=0.5)
i.plt.axhline(0, linestyle="--")
i.plt.xlabel("Predicted quality")
i.plt.ylabel("Residual")
i.plt.title("Residuals vs Predictions")
i.plt.show()