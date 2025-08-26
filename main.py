import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ————— Data —————
from src.data_preprocessing import prepare_data
from src.data_lstm import create_lstm_data

# ————— Models (tablicowe) —————
from src.models.decision_tree import train_decision_tree, predict_decision_tree
from src.models.random_forest import train_random_forest, predict_random_forest
from src.models.mlp import train_mlp, predict_mlp
from src.models.xgboost_model import train_xgboost, predict_xgboost    # RS + ES (final)
from src.models.xgboost_custom import train_xgboost_custom               # (opcjonalnie)

# ————— Models (sekwencyjne) —————
from src.models.lstm import train_lstm, predict_lstm

# ————— Utils (wykresy) —————
from src.utils import plot_predictions, join_prediction_plots, plot_error_heatmap
from src.feature_viz import analyze_permutation_only, analyze_lstm_permutation_only


# =============================================================================
# KONFIGURACJA
# =============================================================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_DIR_IMPORTANCE = os.path.join(RESULTS_DIR, "importance")
os.makedirs(RESULTS_DIR_IMPORTANCE, exist_ok=True)


# Jedna metryka sterująca strojenie DT/RF/XGB: "mae" | "rmse" | "r2" | "multi"
MODEL_METRIC = "mae"

# Co uruchamiać
RUN_BASELINE       = False
RUN_DECISION_TREE  = False
RUN_RANDOM_FOREST  = False
RUN_XGB_RS_ES      = False
RUN_MLP            = True
RUN_LSTM           = False

# =============================================================================
# POMOCNICZE
# =============================================================================
def evaluate_and_log(name: str, y_true: np.ndarray, y_pred: np.ndarray, params) -> dict:
    """Liczy MAE/RMSE/R^2 i oddaje spójny rekord do tabeli wyników."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"model": name, "MAE mean": mae, "RMSE": rmse, "R^2": r2, "Best params": params}

def print_model_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def map_metric_to_dt_refit(metric: str) -> str:
    """
    Mapowanie metryki na klucz refit dla Decision Tree:
    - DT w Twojej wersji ma scoring dict z kluczami:
      "neg_mean_absolute_error", "rmse_custom", "r2".
    """
    if metric == "rmse":
        return "rmse_custom"
    if metric == "r2":
        return "r2"
    # "mae" i "multi" → stabilnie refit po MAE
    return "neg_mean_absolute_error"

# =============================================================================
# 0) DANE – PRESET dla modeli TABLICOWYCH
# ============================================================================

results = []

# =============================================================================
# 1) DRZEWO DECYZYJNE
# =============================================================================
if RUN_DECISION_TREE:
    print_model_header("[Decision Tree] Strojenie + ewaluacja")
    t0 = time.time()

    # — dane specyficzne dla DT —
    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        dataset_type="ta_lags_2",    # ← tu możesz podmieniać na inny preset
        test_size=0.2,
        scale_X=False,               # drzewa nie potrzebują skalowania
        target_mode="close_next"
    )
    close_idx = feature_names.index("Close")
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None

    refit_metric = map_metric_to_dt_refit(MODEL_METRIC)
    target_transform = "delta"       # domyślnie delta

    result_dt, best_dt = train_decision_tree(
        X_train, y_train,
        cv_splits=5, n_iter=120,
        metric=MODEL_METRIC, refit_metric=refit_metric,
        cv_gap=20, fast=False,
        target_transform=target_transform,
        close_idx=close_idx, atr_idx=atr_idx
    )

    y_pred_dt = predict_decision_tree(best_dt, X_test)
    res = evaluate_and_log(result_dt["model"], y_test, y_pred_dt, result_dt.get("Best params","-"))
    results.append(res)

    plot_predictions(y_test, y_pred_dt, dates_test,
        model_name=f"Decision Tree ({'residual' if target_transform else 'level'})",
        save_path=f"{RESULTS_DIR}/decision_tree_plot.png")

    analyze_permutation_only(
        model=best_dt, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=RESULTS_DIR_IMPORTANCE, model_name="Decision Tree",
        use_residual=(target_transform is not None),
        close_idx=close_idx, atr_idx=atr_idx
    )
    
    
# =============================================================================
# 2) RANDOM FOREST
# =============================================================================
if RUN_RANDOM_FOREST:
    print_model_header("[Random Forest] Strojenie + ewaluacja")
    t0 = time.time()

    # — dane specyficzne dla RF —
    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        dataset_type="ta_lags_core",  # np. chudszy preset dla testu
        test_size=0.2,
        scale_X=False,
        target_mode="close_next"
    )
    close_idx = feature_names.index("Close")
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None

    target_transform = "delta"

    result_rf, best_rf = train_random_forest(
        X_train, y_train,
        random_state=42, cv_splits=5, n_iter=60,
        metric=MODEL_METRIC, cv_gap=20, fast=True,
        target_transform=target_transform,
        close_idx=close_idx, atr_idx=atr_idx
    )

    y_pred_rf = predict_random_forest(best_rf, X_test)
    res = evaluate_and_log(result_rf["model"], y_test, y_pred_rf, result_rf["Best params"])
    results.append(res)

    plot_predictions(y_test, y_pred_rf, dates_test,
        model_name=f"Random Forest ({'residual' if target_transform!='level' else 'level'})",
        save_path=f"{RESULTS_DIR}/random_forest_plot.png")

    analyze_permutation_only(
        model=best_rf, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=RESULTS_DIR_IMPORTANCE, model_name="Random Forest",
        use_residual=(target_transform != "level"),
        close_idx=close_idx, atr_idx=atr_idx
    )

# =============================================================================
# 3) XGBOOST – RandomizedSearch + EarlyStopping (final fit na train-val)
# =============================================================================


# =============================================================================
if RUN_XGB_RS_ES:
    print_model_header("[XGBoost] Strojenie (no-ES) → Retrain z EarlyStopping (TSCV) + ewaluacja")
    t0 = time.time()

    # — dane specyficzne dla XGB —
    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        dataset_type="mixed",         # np. mieszany preset dla XGB
        test_size=0.2,
        scale_X=False,                # XGB nie wymaga skalowania
        target_mode="close_next"
    )
    close_idx = feature_names.index("Close")
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None
    target_transform = "delta"       # domyślnie delta

    result_xgb, best_xgb = train_xgboost(
        X_train, y_train,
        cv_splits=5, n_iter=80, val_fraction=0.2, early_stopping_rounds=200,
        metric=MODEL_METRIC, cv_gap=20, fast=True,
        close_idx=close_idx, atr_idx=atr_idx,
        target_transform=target_transform
    )

    y_pred_xgb = predict_xgboost(best_xgb, X_test)
    res = evaluate_and_log(result_xgb["model"], y_test, y_pred_xgb, result_xgb["Best params"])
    results.append(res)

    plot_predictions(y_test, y_pred_xgb, dates_test,
        model_name=f"XGBoost (Tuning→ES, {target_transform})",
        save_path=f"{RESULTS_DIR}/xgboost_rs_es_plot.png")

    analyze_permutation_only(
        model=best_xgb, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=RESULTS_DIR_IMPORTANCE, model_name="XGBoost",
        use_residual=(target_transform != "level"),
        close_idx=close_idx,
        atr_idx=atr_idx if target_transform == "delta_over_atr" else None
    )

# =============================================================================
# 4) MLP (Grid/RandomizedSearchCV w pipeline + scaling)
# =============================================================================
if RUN_MLP:
    print_model_header("[MLP] Strojenie + ewaluacja")
    t0 = time.time()

    # — dane specyficzne dla MLP —
    # Uwaga: w pipeline MLP masz StandardScaler, więc tutaj NIE skaluj (scale_X=False),
    # żeby nie robić podwójnego skalowania.
    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        dataset_type="ta_lags_2",
        test_size=0.2,
        scale_X=False,                # scaler jest w pipeline
        target_mode="close_next"
    )

    result_mlp, best_mlp = train_mlp(
        X_train, y_train,
        random_state=42, cv_splits=5, n_iter=30,
        fast=True, cv_gap=20
    )

    y_pred_mlp = predict_mlp(best_mlp, X_test)
    res = evaluate_and_log(result_mlp["model"], y_test, y_pred_mlp, result_mlp["Best params"])
    results.append(res)

    plot_predictions(y_test, y_pred_mlp, dates_test,
        model_name="MLPRegressor", save_path=f"{RESULTS_DIR}/mlp_plot.png")

    analyze_permutation_only(
        model=best_mlp, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=RESULTS_DIR_IMPORTANCE, model_name="MLP",
        use_residual=False
    )

# =============================================================================
# 5) LSTM (sekwencje)
# =============================================================================
if RUN_LSTM:
    print_model_header("[LSTM] Trening + ewaluacja")

    # 1) Dane sekwencyjne – ten pipeline skaluje TYLKO na train (brak przecieku)
    X_tr_seq, X_te_seq, y_tr_seq, y_te_seq, dates_te_seq, scaler_y, feature_names_seq = create_lstm_data(
    file_path="data/gold_data.csv",
    dataset_type="ta_lags",
    target_mode="close_next",
    window_size=30,
    test_size=0.2,
    scale_X=True
)

    # 2) Trening
    t0 = time.time()
    # Szybkie pętle badawcze → fast=True; do wyników końcowych ustaw fast=False
    model_lstm, history_lstm, _ = train_lstm(
        X_tr_seq, y_tr_seq,
        val_fraction=0.10,        # walidacja = ostatnie 10% train (chronologicznie)
        epochs=60,
        batch_size=64,
        units_1=128,
        units_2=32,
        dropout_rate=0.2,
        lr=1e-3,
        es_patience=8,
        rlrop_patience=4,
        verbose=2,
        fast=True                 # ← przełącz na False do finalnego treningu
    )

    # 3) Predykcja + odtworzenie skali (bo y było standaryzowane)
    y_pred_lstm_scaled = predict_lstm(model_lstm, X_te_seq)
    y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).ravel()
    y_test_inv  = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()

    # 4) Ewaluacja + wykres
    params_str = "LSTM(units=(128,32), dropout=0.2, Adam lr=1e-3, fast=True, ES+RLROP)"
    res = evaluate_and_log("LSTM", y_test_inv, y_pred_lstm, params_str)
    results.append(res)

    plot_predictions(
        y_test_inv, y_pred_lstm, dates_te_seq,
        model_name="LSTM",
        save_path=f"{RESULTS_DIR}/lstm_plot.png"
    )

    print(f"[LSTM] Czas: {time.time() - t0:.2f}s | "
          f"MAE={res['MAE mean']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R^2']:.5f}")
    
    y_pred_lstm_scaled = predict_lstm(model_lstm, X_te_seq)

# permutation importance dla LSTM – użyj skalowanego y_test (y_te_seq):
    analyze_lstm_permutation_only(
        model=model_lstm,
        X_test_seq=X_te_seq,
        y_test_scaled=y_te_seq,          # ← ważne: SKALOWANE y!
        feature_names=feature_names_seq,      # z create_lstm_data()
        results_dir=RESULTS_DIR_IMPORTANCE if 'RESULTS_DIR_IMPORTANCE' in globals() else RESULTS_DIR,
        model_name="LSTM",
        n_repeats=10,
)
# =============================================================================
# ZBIORCZA TABELA WYNIKÓW + HEATMAPA BŁĘDÓW
# =============================================================================
pd.set_option("display.float_format", "{:.5f}".format)  # 5 miejsc, m.in. dla R² w wydruku
results_df = pd.DataFrame(results)

# zapis do CSV – również z 5 miejscami po przecinku
results_df.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False, float_format="%.5f")

print("\n== PODSUMOWANIE ==")
print(results_df.to_string(index=False,
                           formatters={"MAE mean": "{:.3f}".format,
                                       "RMSE": "{:.3f}".format,
                                       "R^2": "{:.5f}".format}))

