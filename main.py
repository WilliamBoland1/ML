import ridge_regression_red as rrr
import Polinomial_regression_MSE as prm
import Pol_pluss_ridge as ppr
import lasso as l
import plotts as p

import importingfile as i

df = i.pd.read_csv("data/winequality-red.csv", sep=";")  # or white

y_true = df["quality"]

y_test_rrr, y_pred_rrr = rrr.ridge_regression(df)

degree = 2

X_test_poly_scaled, y_test_prm, y_pred_prm = prm.polynomial_regression(df,degree)

y_test_ppr, y_pred_ppr, best_params_ppr = ppr.poly_ridge_regression(df)

y_test_lasso, y_pred_lasso, lasso_model = l.lasso_regression(df)


residuals_rrr = y_test_rrr - y_pred_rrr
residuals_prm = y_test_prm - y_pred_prm
residuals_ppr = y_test_ppr - y_pred_ppr
residuals_l = y_test_lasso - y_pred_lasso

p.plot_residual(residuals_rrr,y_pred_rrr,"Ridge Regression")
p.plot_residual(residuals_prm,y_pred_prm,"Polinomial Regression")
p.plot_residual(residuals_ppr,y_pred_ppr,"Polinomial + Ridge Regression")
p.plot_residual(residuals_l,y_pred_lasso,"Lasso Regression")

i.plt.show()