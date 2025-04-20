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

X_train, X_test, y_train, y_test, dates_test = prepare_data()
results = []
model_names = []
mae_values = []

# Decision Tree
result_dt, best_model_dt = train_decision_tree(X_train, y_train)
results.append(result_dt)
y_pred_dt = predict_decision_tree(best_model_dt, X_test)
plot_predictions(y_test, y_pred_dt, dates_test, model_name="Decision Tree", save_path="results/decision_tree_plot.png")
model_names.append("Decision Tree (GridSearch)")
mae_values.append(result_dt['MAE mean'])

# Random Forest
result_rf, best_model_rf = train_random_forest(X_train, y_train)
results.append(result_rf)
y_pred_rf = predict_random_forest(best_model_rf, X_test)
plot_predictions(y_test, y_pred_rf, dates_test, model_name="Random Forest", save_path="results/random_forest_plot.png")
model_names.append("Random Forest (GridSearch)")
mae_values.append(result_rf['MAE mean'])

# SVM
result_svm, best_model_svm = train_svm(X_train, y_train)
results.append(result_svm)
y_pred_svm = predict_svm(best_model_svm, X_test)
plot_predictions(y_test, y_pred_svm, dates_test, model_name="SVM", save_path="results/svm_plot.png")
model_names.append("SVM (SVR, GridSearch)")
mae_values.append(result_svm['MAE mean'])

# XGBoost
result_xgb, best_model_xgb = train_xgboost(X_train, y_train)
results.append(result_xgb)
y_pred_xgb = predict_xgboost(best_model_xgb, X_test)
plot_predictions(y_test, y_pred_xgb, dates_test, model_name="XGBoost", save_path="results/xgboost_plot.png")
model_names.append("XGBoost (GridSearch)")
mae_values.append(result_xgb['MAE mean'])

# XGBoost Early Stopping
result_xgb_es, best_model_xgb_es, y_pred_xgb_es = train_xgboost_custom(X_train, X_test, y_train, y_test)
results.append(result_xgb_es)
plot_predictions(y_test, y_pred_xgb_es, dates_test, model_name="XGBoost (Early Stop)", save_path="results/xgboost_early_plot.png")
model_names.append("XGBoost (Early Stopping)")
mae_values.append(result_xgb_es['MAE mean'])

# MLP
start_time = time.time()
result_mlp, best_model_mlp = train_mlp(X_train, y_train)
results.append(result_mlp)
y_pred_mlp = predict_mlp(best_model_mlp, X_test)
plot_predictions(y_test, y_pred_mlp, dates_test, model_name="MLPRegressor", save_path="results/mlp_plot.png")
model_names.append("MLPRegressor (GridSearch, scaled)")
mae_values.append(result_mlp['MAE mean'])
end_time = time.time()
print(f"Czas wykonania MLP: {end_time - start_time:.2f} sekund")

# LSTM
X_train, X_test, y_train, y_test, dates_test, scaler_y = create_lstm_data()
start_time = time.time()
model_lstm, history_lstm = train_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
y_pred_lstm_scaled = predict_lstm(model_lstm, X_test)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
end_time = time.time()
mae_lstm = mean_absolute_error(y_test_inv, y_pred_lstm)
results.append({
    'model': 'LSTM',
    'MAE mean': mae_lstm,
    'Best params': 'custom (LSTM(64), dropout=0.2, adam)'
})
model_names.append("LSTM")
mae_values.append(mae_lstm)
plot_predictions(y_test_inv, y_pred_lstm, dates_test, model_name="LSTM", save_path="results/lstm_plot.png")
print(f"Czas wykonania LSTM: {end_time - start_time:.2f} sekund")

# Zapis tabeli wyników
results_df = pd.DataFrame(results)
results_df.to_csv('results/comparison_table.csv', index=False)
print(results_df)

# Zbiorczy wykres PNG
join_prediction_plots()

# Heatmapa MAE na podstawie wyników
df_mae = pd.DataFrame({"Model": model_names, "MAE": mae_values}).set_index("Model").sort_values("MAE")
plt.figure(figsize=(10, 6))
sns.heatmap(df_mae, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, cbar_kws={'label': 'MAE'})
plt.title('Heatmapa MAE dla modeli regresyjnych (posortowana)')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("results/models_mae_heatmap_sorted.png")
plt.close()