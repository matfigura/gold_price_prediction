from src.data_preprocessing import prepare_data
from src.models.decision_tree import train_decision_tree, predict_decision_tree
from src.models.random_forest import train_random_forest, predict_random_forest
# from src.models.random_forest_randomized import train_random_forest_randomized, predict_rf_randomized
from src.models.mlp import train_mlp, predict_mlp
from src.models.svm import train_svm, predict_svm
from src.models.xgboost import train_xgboost, predict_xgboost
from src.models.xgboost_custom import train_xgboost_custom
from src.models.lstm import train_lstm, predict_lstm
from src.data_lstm import create_lstm_data
from src.utils import plot_predictions, join_prediction_plots
from sklearn.metrics import mean_absolute_error
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image
import numpy as np
from src.utils import plot_error_heatmap

X_train, X_test, y_train, y_test, dates_test = prepare_data()
results = []
model_names = []
mae_values = []

# --- Decision Tree ---
result_dt, best_model_dt = train_decision_tree(X_train, y_train)
y_pred_dt = predict_decision_tree(best_model_dt, X_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
results.append({
    'model': 'Decision Tree (GridSearch)',
    'MAE mean': mae_dt,
    'MSE': mse_dt,
    'RMSE': rmse_dt,
    'Best params': result_dt['Best params']
})
plot_predictions(y_test, y_pred_dt, dates_test, model_name="Decision Tree", save_path="results/decision_tree_plot.png")

# --- Random Forest ---
result_rf, best_model_rf = train_random_forest(X_train, y_train)
y_pred_rf = predict_random_forest(best_model_rf, X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
results.append({
    'model': 'Random Forest (GridSearch)',
    'MAE mean': mae_rf,
    'MSE': mse_rf,
    'RMSE': rmse_rf,
    'Best params': result_rf['Best params']
})
plot_predictions(y_test, y_pred_rf, dates_test, model_name="Random Forest", save_path="results/random_forest_plot.png")

'''# Tylko Random Forest z RandomizedSearch
result_rf_rand, best_model_rf_rand = train_random_forest_randomized(X_train, y_train)
results.append(result_rf_rand)
y_pred_rf_rand = predict_rf_randomized(best_model_rf_rand, X_test)
plot_predictions(y_test, y_pred_rf_rand, dates_test, model_name="Random Forest (Randomized)", save_path="results/random_forest_randomized_plot.png")

# --- SVM ---
result_svm, best_model_svm = train_svm(X_train, y_train)
y_pred_svm = predict_svm(best_model_svm, X_test)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)
rmse_svm = np.sqrt(mse_svm)
results.append({
    'model': 'SVM (SVR, GridSearch)',
    'MAE mean': mae_svm,
    'MSE': mse_svm,
    'RMSE': rmse_svm,
    'Best params': result_svm['Best params']
})
plot_predictions(y_test, y_pred_svm, dates_test, model_name="SVM", save_path="results/svm_plot.png")'''

# --- XGBoost ---
result_xgb, best_model_xgb = train_xgboost(X_train, y_train)
y_pred_xgb = predict_xgboost(best_model_xgb, X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
results.append({
    'model': 'XGBoost (GridSearch)',
    'MAE mean': mae_xgb,
    'MSE': mse_xgb,
    'RMSE': rmse_xgb,
    'Best params': result_xgb['Best params']
})
plot_predictions(y_test, y_pred_xgb, dates_test, model_name="XGBoost", save_path="results/xgboost_plot.png")

# --- XGBoost Early Stopping ---
result_xgb_es, best_model_xgb_es, y_pred_xgb_es = train_xgboost_custom(X_train, X_test, y_train, y_test)
mae_xgb_es = mean_absolute_error(y_test, y_pred_xgb_es)
mse_xgb_es = mean_squared_error(y_test, y_pred_xgb_es)
rmse_xgb_es = np.sqrt(mse_xgb_es)
results.append({
    'model': 'XGBoost (Early Stopping)',
    'MAE mean': mae_xgb_es,
    'MSE': mse_xgb_es,
    'RMSE': rmse_xgb_es,
    'Best params': result_xgb_es['Best params']
})
plot_predictions(y_test, y_pred_xgb_es, dates_test, model_name="XGBoost (Early Stop)", save_path="results/xgboost_early_plot.png")

# --- MLP ---
start_time = time.time()
result_mlp, best_model_mlp = train_mlp(X_train, y_train)
y_pred_mlp = predict_mlp(best_model_mlp, X_test)
end_time = time.time()
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
results.append({
    'model': 'MLPRegressor (GridSearch, scaled)',
    'MAE mean': mae_mlp,
    'MSE': mse_mlp,
    'RMSE': rmse_mlp,
    'Best params': result_mlp['Best params']
})
plot_predictions(y_test, y_pred_mlp, dates_test, model_name="MLPRegressor", save_path="results/mlp_plot.png")
print(f"Czas wykonania MLP: {end_time - start_time:.2f} sekund")

# --- LSTM ---
from src.data_lstm import create_lstm_data
X_train, X_test, y_train, y_test, dates_test, scaler_y = create_lstm_data()
start_time = time.time()
model_lstm, history_lstm = train_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
y_pred_lstm_scaled = predict_lstm(model_lstm, X_test)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
end_time = time.time()
mae_lstm = mean_absolute_error(y_test_inv, y_pred_lstm)
mse_lstm = mean_squared_error(y_test_inv, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
results.append({
    'model': 'LSTM',
    'MAE mean': mae_lstm,
    'MSE': mse_lstm,
    'RMSE': rmse_lstm,
    'Best params': 'custom (LSTM(64), dropout=0.2, adam)'
})
plot_predictions(y_test_inv, y_pred_lstm, dates_test, model_name="LSTM", save_path="results/lstm_plot.png")
print(f"Czas wykonania LSTM: {end_time - start_time:.2f} sekund")


# Zapis tabeli wyników
results_df = pd.DataFrame(results)
results_df.to_csv('results/comparison_table.csv', index=False)
print(results_df)

# Zbiorczy wykres PNG
join_prediction_plots()


# Automatyczne wyświetlenie heatmapy
Image.open("results/error_metrics_heatmap.png").show()
plot_error_heatmap(results_df)

importances = best_model_rf.feature_importances_
feature_names = [
    "Open", "High", "Low", "Volume", 
    "return", "rolling_mean_5", "rolling_std_5", 
    "high_low_diff", "close_open_ratio", "days_since_max"
]

sorted_idx = importances.argsort()
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances)
plt.xlabel("Ważność cechy")
plt.title("Random Forest – Feature Importances")
plt.tight_layout()
plt.savefig("results/feature_importance_rf.png")
plt.show()