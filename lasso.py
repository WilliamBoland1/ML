import utils.importingfile as i

def lasso_regression(df, random_state=42):
    # Separate features (X) and target variable (y)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Train / test split
    X_train, X_test, y_train, y_test = i.train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Standardize features (required for Lasso)
    scaler = i.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define alpha values to test
    alphas = [0.001, 0.01, 0.1, 1, 10]

    # Initialize Lasso model
    lasso = i.Lasso(max_iter=10000, random_state=random_state)

    # Grid search for optimal alpha
    param_grid = {"alpha": alphas}

    grid = i.GridSearchCV(
        lasso,
        param_grid,
        cv=i.KFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring="neg_mean_squared_error"
    )

    grid.fit(X_train_scaled, y_train)

    # Best model
    best_lasso = grid.best_estimator_

    # Predict on test set
    y_pred = best_lasso.predict(X_test_scaled)
    for index in range(len(y_pred)):
        if y_pred[index] < 1:
            y_pred[index] = 1
        elif y_pred[index] > 10:
            y_pred[index]  = 10

    return y_test, y_pred, best_lasso
