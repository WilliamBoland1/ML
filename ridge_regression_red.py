import importingfile as i  # Import helper module that wraps sklearn utilities

def ridge_regression(df):
    # Separate features (X) and target variable (y)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Split the dataset into training and test sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = i.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize a standard scaler to normalize features
    scaler = i.StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaling parameters to the test data
    X_test_scaled = scaler.transform(X_test)

    # Initialize Ridge regression with an initial regularization strength
    ridge = i.Ridge(alpha=1.0)  # initial guess

    # Train the Ridge regression model
    ridge.fit(X_train_scaled, y_train)

    # Predict target values for the test set
    y_pred = ridge.predict(X_test_scaled)
    for index in range(len(y_pred)):
        if y_pred[index] < 1:
            y_pred[index] = 1
        elif y_pred[index] > 10:
            y_pred[index]  = 10

    # Compute evaluation metrics to assess model performance
    rmse = i.root_mean_squared_error(y_test, y_pred)  # Root Mean Squared Error
    mae = i.mean_absolute_error(y_test, y_pred)       # Mean Absolute Error
    r2 = i.r2_score(y_test, y_pred)                    # R-squared score

    # Print evaluation results
    # print(f"RMSE: {rmse:.3f}")
    # print(f"MAE: {mae:.3f}")
    # print(f"R²: {r2:.3f}")

    # Define a range of alpha values to test for regularization strength
    alphas = [0.01, 0.1, 1, 10, 100]

    # Initialize a new Ridge regression model (alpha will be tuned)
    ridge = i.Ridge()

    # Create a parameter grid for GridSearchCV
    param_grid = {"alpha": alphas}

    # Set up grid search with 5-fold cross-validation
    # Negative MSE is used because GridSearchCV maximizes the scoring metric
    grid = i.GridSearchCV(
        ridge,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    # Perform grid search on the training data
    grid.fit(X_train_scaled, y_train)

    # Print the best alpha value found by cross-validation
    #print("Best alpha:", grid.best_params_["alpha"])

    # Retrieve the best Ridge model from the grid search
    best_ridge = grid.best_estimator_

    # Predict again using the optimized Ridge model
    y_pred = best_ridge.predict(X_test_scaled)
    for index in range(len(y_pred)):
        if y_pred[index] < 1:
            y_pred[index] = 1
        elif y_pred[index] > 10:
            y_pred[index]  = 10

    # Return the scaled test features and corresponding predictions
    return y_test, y_pred
