import importingfile as i

def poly_ridge_regression(df, degrees=(1, 2, 3), alphas=(0.01, 0.1, 1, 10, 100)):
    # ----------------------------
    # 1) Split features and target
    # ----------------------------
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = i.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # 2) Grid search over:
    #    - Polynomial degree
    #    - Ridge alpha
    # ----------------------------
    best_model = None
    best_params = None
    best_cv_mse = float("inf")

    for deg in degrees:
        # Build polynomial features
        poly = i.PolynomialFeatures(degree=deg, include_bias=False)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Scale after polynomial expansion (important)
        scaler = i.StandardScaler()
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = scaler.transform(X_test_poly)

        # Tune alpha using CV (MSE)
        ridge = i.Ridge()
        param_grid = {"alpha": list(alphas)}

        grid = i.GridSearchCV(
            ridge,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error"
        )

        grid.fit(X_train_poly_scaled, y_train)

        # GridSearchCV stores negative MSE, so flip sign
        cv_mse = -grid.best_score_

        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_model = (deg, poly, scaler, grid.best_estimator_)
            best_params = {"degree": deg, "alpha": grid.best_params_["alpha"]}

    # ----------------------------
    # 3) Evaluate best model on test
    # ----------------------------
    deg, poly, scaler, ridge_best = best_model

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)

    ridge_best.fit(X_train_poly_scaled, y_train)
    y_pred = ridge_best.predict(X_test_poly_scaled)

    # Metrics
    mse = i.mean_squared_error(y_test, y_pred)
    rmse = i.root_mean_squared_error(y_test, y_pred)
    mae = i.mean_absolute_error(y_test, y_pred)
    r2 = i.r2_score(y_test, y_pred)

    print("Best params:", best_params)
    print(f"CV MSE (best): {best_cv_mse:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Return test labels and predictions for plotting
    return y_test, y_pred, best_params
