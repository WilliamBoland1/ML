import importingfile as i

df = i.pd.read_csv("data/winequality-red.csv", sep=";")  # or white

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = i.train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = i.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


ridge = i.Ridge(alpha=1.0)  # initial guess
ridge.fit(X_train_scaled, y_train)


y_pred = ridge.predict(X_test_scaled)

rmse = i.mean_squared_error(y_test, y_pred, squared=False)
mae = i.mean_absolute_error(y_test, y_pred)
r2 = i.r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")


alphas = [0.01, 0.1, 1, 10, 100]

ridge = i.Ridge()

param_grid = {"alpha": alphas}

grid = i.GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train_scaled, y_train)

print("Best alpha:", grid.best_params_["alpha"])

best_ridge = grid.best_estimator_
y_pred = best_ridge.predict(X_test_scaled)
