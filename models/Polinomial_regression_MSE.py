import utils.importingfile as i

def polynomial_regression(df, degree=2, random_state=42):
    # Split features and target
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Train / test split
    X_train, X_test, y_train, y_test = i.train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Create polynomial features
    poly = i.PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scale features (VERY important for polynomial regression)
    scaler = i.StandardScaler()

    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)

    # Train linear regression on polynomial features
    model = i.LinearRegression()
    model.fit(X_train_poly_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_poly_scaled)
    for index in range(len(y_pred)):
        if y_pred[index] < 1:
            y_pred[index] = 1
        elif y_pred[index] > 10:
            y_pred[index]  = 10


    return X_test_poly_scaled, y_test, y_pred