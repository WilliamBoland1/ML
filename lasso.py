import importingfile as i

def lasso_regression(df):
    # Separate features (X) and target variable (y)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Train / test split
    X_train, X_test, y_train, y_test = i.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (required for Lasso)
    scaler = i.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define alpha values to test
    alphas = [0.001, 0.01, 0.1, 1, 10]

    # Initialize Lasso model
    lasso = i.Lasso(max_iter=10000)

    # Grid search for optimal alpha
    param_grid = {"alpha": alphas}

    grid = i.GridSearchCV(
        lasso,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X_train_scaled, y_train)

    # Best model
    best_lasso = grid.best_estimator_

    # Predict on test set
    y_pred = best_lasso.predict(X_test_scaled)

    # Evaluation metrics
    mse = i.mean_squared_error(y_test, y_pred)
    rmse = i.root_mean_squared_error(y_test, y_pred)
    mae = i.mean_absolute_error(y_test, y_pred)
    r2 = i.r2_score(y_test, y_pred)

    print("Best alpha:", grid.best_params_["alpha"])
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")

    return y_test, y_pred, best_lasso
