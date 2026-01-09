import ridge_regression_red as rrr
import Polinomial_regression_MSE as prm
import plotts as p

import importingfile as i

df = i.pd.read_csv("data/winequality-red.csv", sep=";")  # or white

y_true = df["quality"]

y_test_rrr, y_rrr = rrr.ridge_regression(df)

degree = 2

X_test_poly_scaled, y_test_prm, y_prm = prm.polynomial_regression(df,degree)

residuals_rrr = y_test_rrr - y_rrr
residuals_prm = y_test_prm - y_prm

p.plot_residual(residuals_rrr,y_rrr,"Ridge Regression")
p.plot_residual(residuals_prm,y_prm,"Polinomial Regression")

i.plt.show()