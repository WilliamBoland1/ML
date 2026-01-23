import importingfile as i
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats


df = i.pd.read_csv("../data/winequality-red.csv", sep=";")

y_true = df["quality"].values
mean_quality = y_true.mean()
mode_number = stats.mode(y_true)
mode_number = mode_number.mode
print(mode_number)
# Predict mean for all samples
y_pred_mean = np.full_like(y_true, mean_quality, dtype=float)
y_pred_mode = np.full_like(y_true, mode_number, dtype=float)

# Metrics
mae = mean_absolute_error(y_true, y_pred_mean)
rmse = mean_squared_error(y_true, y_pred_mean, squared=False)
r2 = r2_score(y_true, y_pred_mean)

mae_1 = mean_absolute_error(y_true, y_pred_mode)
rmse_1 = mean_squared_error(y_true, y_pred_mode, squared=False)
r2_1 = r2_score(y_true, y_pred_mode)

# Within ±1 of the mean
within_pm1 = np.abs(y_true - mean_quality) <= 1
within_pm2 = np.abs(y_true - mode_number) <= 1
fraction_within_pm1 = within_pm1.mean()    # fraction
fraction_within_pm2 = within_pm2.mean()
percentage_within_pm1 = fraction_within_pm1 * 100
percentage_within_pm2 = fraction_within_pm2 * 100

print("Mean Predictor")
print(f"Mean quality: {mean_quality:.2f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
print(f"Within ±1 of mean: {percentage_within_pm1:.2f}%")
print("Mode Predictor")
print(f"Mode quality: {mode_number}")
print(f"MAE:  {mae_1:.4f}")
print(f"RMSE: {rmse_1:.4f}")
print(f"R²:   {r2_1:.4f}")
print(f"Within ±1 of mean: {percentage_within_pm2:.2f}%")
