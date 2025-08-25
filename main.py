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
from src.feature_viz import analyze_model_features


# =============================================================================
# KONFIGURACJA
# =============================================================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Preset danych dla modeli tablicowych: "ohlc" | "ta_lags" | "mixed" | "ta_lags_plus"
DATASET_TYPE = "ta_lags"
# Cel: "close_next" (poziom) lub "return_next" (zwrot log/lin – wg Twojej implementacji)
TARGET_MODE  = "close_next"
# Skalowanie cech: None → auto (dla "ohlc": False, dla innych: True)
SCALE_X      = False

# Jedna metryka sterująca strojenie DT/RF/XGB: "mae" | "rmse" | "r2" | "multi"
MODEL_METRIC = "mae"

# Co uruchamiać
RUN_BASELINE       = False
RUN_DECISION_TREE  = False
RUN_RANDOM_FOREST  = False
RUN_XGB_RS_ES      = True
RUN_MLP            = False
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

def baseline_persistence(y_train: np.ndarray, y_test: np.ndarray):
    """
    Persistence:  ŷ_{t} = y_{t-1}
    Pierwsza predykcja dla testu = ostatnia obserwacja z tren.
    """
    y_pred = np.empty_like(y_test)
    y_pred[0]  = y_train[-1]
    y_pred[1:] = y_test[:-1]
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return y_pred, mae, rmse, r2

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
# =============================================================================
X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
    dataset_type=DATASET_TYPE,
    test_size=0.2,
    scale_X=SCALE_X,
    target_mode=TARGET_MODE,
)
close_idx = feature_names.index("Close")
atr_idx   = feature_names.index("atr_14")   # tylko jeśli preset ma ATR!

results = []

# =============================================================================
# BASELINE: persistence
# =============================================================================
if RUN_BASELINE:
    print_model_header("[Baseline] Persistence (y_{t} = y_{t-1})")
    y_pred_b, mae_b, rmse_b, r2_b = baseline_persistence(y_train, y_test)
    results.append({
        "model": "Baseline (persistence: y_{t}=y_{t-1})",
        "MAE mean": mae_b, "RMSE": rmse_b, "R^2": r2_b, "Best params": "-"
    })
    # (opcjonalnie) wykres:
    # plot_predictions(y_test, y_pred_b, dates_test, model_name="Baseline",
    #                  save_path=f"{RESULTS_DIR}/baseline_plot.png")
    print(f"[Baseline] MAE={mae_b:.3f}  RMSE={rmse_b:.3f}  R²={r2_b:.5f}")

# =============================================================================
# 1) DRZEWO DECYZYJNE
# =============================================================================
if RUN_DECISION_TREE:
    print_model_header("[Decision Tree] Strojenie + ewaluacja")
    t0 = time.time()

    refit_metric = map_metric_to_dt_refit(MODEL_METRIC)

    # indeksy do rekonstrukcji poziomu
    close_idx = feature_names.index("Close")

    # wybór transformacji celu (Δ lub Δ/ATR)
    target_transform = "delta"   # możliwe: "delta" | "delta_over_atr" | None

    # ATR potrzebny tylko gdy target_transform = "delta_over_atr"
    if target_transform == "delta_over_atr":
        try:
            atr_idx = feature_names.index("atr_14")
        except ValueError:
            raise RuntimeError("Wybrano target_transform='delta_over_atr', ale w cechach brak 'atr_14'.")
    else:
        atr_idx = None

    # trenowanie (TimeSeriesSplit z gap=20, bez przecieku; residual learning)
    result_dt, best_dt = train_decision_tree(
        X_train, y_train,
        cv_splits=5,
        n_iter=120,
        metric=MODEL_METRIC,
        refit_metric=refit_metric,
        cv_gap=20,                      # przerwa = max lag w cechach (eliminuje micro-leakage)
        fast=False,                     # na szybkie testy ustaw True
        target_transform=target_transform,
        close_idx=close_idx,
        atr_idx=atr_idx
    )

    # ważne: predykcja przez helper z rekonstrukcją poziomu (Close_t + Δ̂ lub Close_t + ATR*Δ̂)
    y_pred_dt = predict_decision_tree(best_dt, X_test)

    # ewaluacja + podstawowy wykres
    res = evaluate_and_log(result_dt["model"], y_test, y_pred_dt, result_dt.get("Best params", "-"))
    results.append(res)

    plot_predictions(
        y_test, y_pred_dt, dates_test,
        model_name=f"Decision Tree ({'residual' if target_transform else 'level'})",
        save_path=f"{RESULTS_DIR}/decision_tree_plot.png"
    )

    print(f"[Decision Tree] Czas: {time.time() - t0:.2f}s | "
          f"MAE={res['MAE mean']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R^2']:.5f}")

    # ── ANALIZA CECH: wywołanie NA KOŃCU BLOKU (bez „zaśmiecania” maina) ─────────
    # zakładam, że masz wrapper w src/viz/plot.py zgodnie z wcześniejszą propozycją:
    #   def analyze_dt_features(best_dt, X_train, X_test, y_train, y_test, feature_names, RESULTS_DIR,
    #                           *, use_residual: bool, close_idx: int, atr_idx: Optional[int]) -> None: ...
    analyze_model_features(
        model=best_dt,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        results_dir=RESULTS_DIR,
        model_name="Decision Tree",
        use_residual=(target_transform is not None),  # bo uczyłeś Δ lub Δ/ATR
        close_idx=close_idx,
        atr_idx=atr_idx,                               # None jeśli nie używasz Δ/ATR
    )
    
    
# =============================================================================
# 2) RANDOM FOREST
# =============================================================================
if RUN_RANDOM_FOREST:
    print_model_header("[Random Forest] Strojenie + ewaluacja")
    t0 = time.time()

    # indeksy pomocnicze
    close_idx = feature_names.index("Close")
    target_transform = "delta"   # "level" | "delta" | "delta_over_atr" | "return"

    if target_transform == "delta_over_atr":
        try:
            atr_idx = feature_names.index("atr_14")
        except ValueError:
            raise RuntimeError("Wybrano target_transform='delta_over_atr', ale w cechach brak 'atr_14'.")
    else:
        atr_idx = None

    # trening (TSCV z gap=20 jak w DT; fast=True na krótkie testy)
    
    result_rf, best_rf = train_random_forest(
        X_train, y_train,
        random_state=42,
        cv_splits=5,
        n_iter=60,
        metric=MODEL_METRIC,
        cv_gap=20,
        fast=True,
        target_transform=target_transform,
        close_idx=close_idx,
        atr_idx=atr_idx
    )

    # ważne: predykcja przez helper (rekonstrukcja poziomu)
    y_pred_rf = predict_random_forest(best_rf, X_test)

    # ewaluacja + wykres
    res = evaluate_and_log(result_rf["model"], y_test, y_pred_rf, result_rf["Best params"])
    results.append(res)

    plot_predictions(
        y_test, y_pred_rf, dates_test,
        model_name=f"Random Forest ({'residual' if target_transform != 'level' else 'level'})",
        save_path=f"{RESULTS_DIR}/random_forest_plot.png"
    )

    print(f"[Random Forest] Czas: {time.time() - t0:.2f}s | "
          f"MAE={res['MAE mean']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R^2']:.5f}")

    # (opcjonalnie) ta sama analiza cech co dla DT:
    from src.viz.plots import analyze_model_features
    analyze_model_features(
        model=best_rf,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        results_dir=RESULTS_DIR,
        model_name="Random Forest",
        use_residual=(target_transform != "level"),
        close_idx=close_idx,
        atr_idx=atr_idx
    )

# =============================================================================
# 3) XGBOOST – RandomizedSearch + EarlyStopping (final fit na train-val)
# =============================================================================
try:
    close_idx = feature_names.index("Close")
except ValueError:
    close_idx = None  # jeśli w danym presecie nie ma 'Close', ustaw use_residual_target=False

# =============================================================================
if RUN_XGB_RS_ES:
    print_model_header("[XGBoost] Strojenie (no-ES) → Retrain z EarlyStopping (TSCV) + ewaluacja")
    t0 = time.time()

    # indeksy pomocnicze
    close_idx = feature_names.index("Close")
    # jeśli w presetcie masz 'atr_14' – możesz użyć delta_over_atr:
    atr_idx = feature_names.index("atr_14") if "atr_14" in feature_names else None

    # wybór transformacji celu (jeśli nie masz ATR w cechach → użyj 'delta')
    target_transform = "delta_over_atr" if atr_idx is not None else "delta"

    result_xgb, best_xgb = train_xgboost(
        X_train, y_train,
        cv_splits=5,
        n_iter=80,
        val_fraction=0.2,
        early_stopping_rounds=200,
        metric=MODEL_METRIC,
        cv_gap=20,
        fast=True,                      # szybkie testy; do finalu: False
        close_idx=close_idx,
        atr_idx=atr_idx,
        target_transform=target_transform
    )

    y_pred_xgb = predict_xgboost(best_xgb, X_test)
    res = evaluate_and_log(result_xgb["model"], y_test, y_pred_xgb, result_xgb["Best params"])
    results.append(res)

    plot_predictions(
        y_test, y_pred_xgb, dates_test,
        model_name=f"XGBoost (Tuning→ES, {target_transform})",
        save_path=f"{RESULTS_DIR}/xgboost_rs_es_plot.png"
    )

    print(f"[XGBoost Tuning→ES] Czas: {time.time() - t0:.2f}s | "
          f"MAE={res['MAE mean']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R^2']:.5f}")

    # (opcjonalnie) baseline i „skill” na Δ
    y_pred_naive = X_test[:, close_idx]
    rmse_naive = np.sqrt(np.mean((y_test - y_pred_naive)**2))
    rmse_model = np.sqrt(np.mean((y_test - y_pred_xgb)**2))
    print(f"[Baseline check] RMSE naive={rmse_naive:.3f} | RMSE model={rmse_model:.3f}")

    delta_true = y_test - X_test[:, close_idx]
    delta_pred = y_pred_xgb - X_test[:, close_idx]
    mae_delta  = np.mean(np.abs(delta_true - delta_pred))
    rmse_delta = np.sqrt(np.mean((delta_true - delta_pred)**2))
    std_delta  = np.std(delta_true)
    print(f"[Delta skill] Δ-MAE={mae_delta:.3f} | Δ-RMSE={rmse_delta:.3f} | std(Δ_true)={std_delta:.3f}")

    
analyze_model_features(
    model=best_xgb,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names,
    results_dir=RESULTS_DIR,
    model_name="XGBoost",
    # Uczysz na delcie (delta_over_atr), więc analizę też licz w tej przestrzeni
    use_residual=(target_transform != "level"),
    close_idx=close_idx,
    atr_idx=atr_idx if target_transform == "delta_over_atr" else None,
)

# =============================================================================
# 4) MLP (Grid/RandomizedSearchCV w pipeline + scaling)
# =============================================================================
if RUN_MLP:
    print_model_header("[MLP] Strojenie + ewaluacja")
    t0 = time.time()
    result_mlp, best_mlp = train_mlp(X_train, y_train)
    y_pred_mlp = predict_mlp(best_mlp, X_test)
    res = evaluate_and_log(result_mlp["model"], y_test, y_pred_mlp, result_mlp["Best params"])
    results.append(res)
    plot_predictions(y_test, y_pred_mlp, dates_test,
                     model_name="MLPRegressor",
                     save_path=f"{RESULTS_DIR}/mlp_plot.png")
    print(f"[MLP] Czas: {time.time() - t0:.2f}s | MAE={res['MAE mean']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R^2']:.5f}")

# =============================================================================
# 5) LSTM (sekwencje)
# =============================================================================
if RUN_LSTM:
    print_model_header("[LSTM] Trening + ewaluacja")
    # LSTM dostaje swój sekwencyjny pipeline (okna, chrono split i skalery w środku)
    X_tr_seq, X_te_seq, y_tr_seq, y_te_seq, dates_te_seq, scaler_y, _ = create_lstm_data()

    t0 = time.time()
    model_lstm, history_lstm, _ = train_lstm(
        X_tr_seq, y_tr_seq,
        val_fraction=0.1,   # walidacja = ostatnie 10% train
        epochs=60, batch_size=64,
        units_1=128, units_2=32, dropout_rate=0.2, lr=1e-3,
        es_patience=8, rlrop_patience=4,
        verbose=2
    )
    y_pred_lstm_scaled = predict_lstm(model_lstm, X_te_seq)
    y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).ravel()
    y_test_inv  = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()

    res = evaluate_and_log("LSTM", y_test_inv, y_pred_lstm,
                           "custom (LSTM(128,32), dropout=0.2, Adam lr=1e-3, ES+RLROP)")
    results.append(res)
    plot_predictions(y_test_inv, y_pred_lstm, dates_te_seq,
                     model_name="LSTM",
                     save_path=f"{RESULTS_DIR}/lstm_plot.png")
    print(f"[LSTM] Czas: {time.time() - t0:.2f}s | MAE={res['MAE mean']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R^2']:.5f}")

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

# Heatmapa korzysta z surowych wartości (wewnątrz funkcji nie zaokrąglaj R²),
# a tu wymuszamy porządek kolumn i ich nazwy jak w Twoim helperze.
plot_error_heatmap(results_df)

# (opcjonalnie) sklej wykresy predykcji w jedną planszę
join_prediction_plots()