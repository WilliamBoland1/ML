import models.ridge_regression as rr
import models.Polinomial_regression_MSE as prm
import models.Pol_pluss_ridge as ppr
import models.lasso as l
import plotts as p
import models.random_forest_regression as rfr
import models.forest_gradient_boost as fgb

import utils.evaluation as ev

import utils.importingfile as i


def main():
    # Load data
    df = i.pd.read_csv("data/winequality-red.csv", sep=";")  # or white
    y_true = df["quality"]  # not used directly, but fine to keep

    # ----------------------- Models -----------------------------

    # Ridge
    y_test_rrr, y_pred_rrr = rr.ridge_regression(df)

    # Polynomial regression
    X_test_poly_scaled, y_test_prm, y_pred_prm = prm.polynomial_regression(df, degree = 2)

    # Polynomial + Ridge
    y_test_ppr, y_pred_ppr, best_params = ppr.poly_ridge_regression(df)

    # Lasso
    y_test_lasso, y_pred_lasso, lasso_model = l.lasso_regression(df)

    # Random Forest
    rf_results = rfr.random_forest_regressor(df, do_plot=False)

    #Gradient boost
    gb_results = fgb.catboost_regression(df, do_plot=False)

    # ----------------------- Evaluation & Metrics -----------------------------
    models_to_evaluate = {
        # "Ridge": (y_test_rrr, y_pred_rrr),
        # "Polynomial": (y_test_prm, y_pred_prm),
        "Poly+Ridge": (y_test_ppr, y_pred_ppr),
        # "Lasso": (y_test_lasso, y_pred_lasso),
        "Random Forest": (rf_results["y_test"], rf_results["y_pred"]),
        # "CatBoost": (gb_results["y_test"], gb_results["y_pred"]),
    }
    
    all_metrics = ev.evaluate_all_models(models_to_evaluate)

    # ----------------------- Residuals -----------------------------
    residuals_rrr = i.np.array(y_test_rrr) - i.np.array(y_pred_rrr)
    residuals_prm = i.np.array(y_test_prm) - i.np.array(y_pred_prm)
    residuals_ppr = i.np.array(y_test_ppr) - i.np.array(y_pred_ppr)
    residuals_l = i.np.array(y_test_lasso) - i.np.array(y_pred_lasso)
    residuals_rf = i.np.array(rf_results["y_test"]) - i.np.array(rf_results["y_pred"])
    residuals_gb = i.np.array(gb_results["y_test"]) - i.np.array(gb_results["y_pred"])

    # ----------------------- Plotting: Residual scatter -----------------------------

    # NOTE: x-axis should be predictions for "Residuals vs Predictions"
    # p.plot_residual(residuals_rrr, y_pred_rrr, "Ridge Regression")
    # p.plot_residual(residuals_prm, y_pred_prm, "Polinomial Regression")
    p.plot_residual(residuals_ppr, y_pred_ppr, "Polinomial + Ridge Regression")
    # p.plot_residual(residuals_l, y_pred_lasso, "Lasso Regression")
    p.plot_residual(residuals_rf, rf_results["y_pred"], "Random Forest")
    # p.plot_residual(residuals_gb, gb_results["y_pred"], "CatBoost")
    i.plt.show()

    #----------------------- Plotting: Predicted vs True -----------------------------

    p.plot_pred_vs_true([
        # ("Ridge", y_test_rrr, y_pred_rrr),
        # ("Poly", y_test_prm, y_pred_prm),
        ("Poly+Ridge", y_test_ppr, y_pred_ppr),
        # ("Lasso", y_test_lasso, y_pred_lasso),
        ("Random Forest", rf_results["y_test"], rf_results["y_pred"]),
        # ("CatBoost", gb_results["y_test"], gb_results["y_pred"]),
    ])
    i.plt.show()

    # ----------------------- Plotting: Residual histogram -----------------------------

    p.plot_residual_hist([
        # ("Ridge", y_test_rrr, y_pred_rrr),
        # ("Poly", y_test_prm, y_pred_prm),
        ("Poly+Ridge", y_test_ppr, y_pred_ppr),
        # ("Lasso", y_test_lasso, y_pred_lasso),
        ("Random Forest", rf_results["y_test"], rf_results["y_pred"]),
        # ("CatBoost", gb_results["y_test"], gb_results["y_pred"]),
    ])
    i.plt.show()

    # ----------------------- Plotting: MAE by quality level -----------------------------

    p.plot_mae_by_quality([
        # ("Ridge", y_test_rrr, y_pred_rrr),
        # ("Poly", y_test_prm, y_pred_prm),
        ("Poly+Ridge", y_test_ppr, y_pred_ppr),
        # ("Lasso", y_test_lasso, y_pred_lasso),
        ("Random Forest", rf_results["y_test"], rf_results["y_pred"]),
        # ("CatBoost", gb_results["y_test"], gb_results["y_pred"]),
    ])
    i.plt.show()

    # ----------------------- Plotting: Coefficients -----------------------------

    feature_names = df.drop("quality", axis=1).columns.tolist()

    # Lasso coefficients (you DO have lasso_model)
    # p.plot_top_coefficients(lasso_model, feature_names, title="Lasso: Top coefficients")

    # Ridge coefficients:
    # You currently do NOT have the ridge model object returned from rrr.ridge_regression(df).
    # If you modify ridge_regression to also return the trained model (best_ridge),
    # then you can plot it here too.
    #
    # Example (if your ridge function returns: y_test, y_pred, ridge_model):
    # y_test_rrr, y_pred_rrr, ridge_model = rrr.ridge_regression(df)
    # p.plot_top_coefficients(ridge_model, feature_names, title="Ridge: Top coefficients")
    #
    i.plt.show()


if __name__ == "__main__":
    main()
