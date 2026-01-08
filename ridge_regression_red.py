import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/winequality-red.csv", sep=";")  # or white

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


ridge = Ridge(alpha=1.0)  # initial guess
ridge.fit(X_train_scaled, y_train)


y_pred = ridge.predict(X_test_scaled)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")


alphas = [0.01, 0.1, 1, 10, 100]

ridge = Ridge()

param_grid = {"alpha": alphas}

grid = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train_scaled, y_train)

print("Best alpha:", grid.best_params_["alpha"])

best_ridge = grid.best_estimator_
y_pred = best_ridge.predict(X_test_scaled)
