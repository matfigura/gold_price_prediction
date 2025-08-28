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
from src.viz.plots import plot_keras_curves


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

def _fmt(secs: float) -> str:
    m, s = divmod(int(round(secs)), 60)
    h, m = divmod(m, 60)
    return (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")

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

    # ——— ustawienia ———
    preset = "mixed"                 # ← zmieniaj tutaj
    target_transform = "delta"       # None => 'level', "delta" => tryb rezydualny
    cv_splits_dt = 5
    n_iter_dt    = 120               # liczba kandydatów w RandomizedSearch

    # ——— dane ———
    X_train, X_test, y_train, y_test, dates_test, feature_names_from_prepare, _ = prepare_data(
        dataset_type=preset,
        test_size=0.2,
        scale_X=False,
        target_mode="close_next"
    )

    # pobierz nazwy cech niezależnie od typu zwracanego przez prepare_data
    if hasattr(X_train, "columns"):
        feature_names = list(X_train.columns)
    else:
        feature_names = list(feature_names_from_prepare)

    assert X_train.shape[1] == len(feature_names), "feature_names != liczba kolumn X_train"
    assert X_test.shape[1]  == len(feature_names), "feature_names != liczba kolumn X_test"

    # indeksy pomocnicze
    close_idx = feature_names.index("Close") if "Close" in feature_names else None
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None

    # ——— trening (z pomiarem czasu strojenia) ———
    refit_metric = map_metric_to_dt_refit(MODEL_METRIC)

    t_tune0 = time.time()
    result_dt, best_dt = train_decision_tree(
        X_train, y_train,
        cv_splits=cv_splits_dt, n_iter=n_iter_dt,
        metric=MODEL_METRIC, refit_metric=refit_metric,
        cv_gap=20, fast=False,
        target_transform=target_transform,
        close_idx=close_idx, atr_idx=atr_idx
    )
    t_tune = time.time() - t_tune0

    # czytelny log strojenia
    fits_total = cv_splits_dt * n_iter_dt           # ~tyle fitów w CV
    def _fmt(secs: float) -> str:
        m, s = divmod(int(round(secs)), 60)
        h, m = divmod(m, 60)
        return (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")
    print(f"[DT] Czas strojenia: {_fmt(t_tune)}  |  CV: {cv_splits_dt}-fold  |  kandydaci: {n_iter_dt}  "
          f"|  ~{fits_total} fitów + refit")

    assert getattr(best_dt, "n_features_in_", None) == X_train.shape[1], \
        "Decision Tree: n_features_in_ != liczbie kolumn X_train"

    # ——— predykcja + logi ———
    y_pred_dt = predict_decision_tree(best_dt, X_test)
    res = evaluate_and_log(result_dt["model"], y_test, y_pred_dt, result_dt.get("Best params", "-"))
    results.append(res)

    # ——— zapisy wykresów (z nazwą presetu) ———
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_label = f"Decision Tree – {preset} ({'residual/delta' if target_transform else 'level'})"
    plot_predictions(
        y_test, y_pred_dt, dates_test,
        model_name=model_label,
        save_path=os.path.join(RESULTS_DIR, f"decision_tree_plot_{preset}.png")
    )

    # Permutation importance – osobny katalog per preset
    perm_dir = os.path.join(RESULTS_DIR_IMPORTANCE, preset)
    os.makedirs(perm_dir, exist_ok=True)
    analyze_permutation_only(
        model=best_dt,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        results_dir=perm_dir,
        model_name=f"Decision Tree – {preset}",
        use_residual=(target_transform is not None),
        close_idx=close_idx, atr_idx=atr_idx
    )
    
    
# =============================================================================
# 2) RANDOM FOREST
# =============================================================================
if RUN_RANDOM_FOREST:
    print_model_header("[Random Forest] Strojenie + ewaluacja")
    t0 = time.time()

    # ——— ustawienia ———
    preset = "ta_research_set_RF" # ohlc, ta_research_set, mixed 
    target_transform = "delta" # level, delta
    cv_splits_rf = 5
    n_iter_rf    = 30
    fast_rf      = False

    # ——— dane ———
    X_train, X_test, y_train, y_test, dates_test, feature_names_from_prepare, _ = prepare_data(
        dataset_type=preset, test_size=0.2, scale_X=False, target_mode="close_next"
    )
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else list(feature_names_from_prepare)
    assert X_train.shape[1] == len(feature_names)
    assert X_test.shape[1]  == len(feature_names)

    close_idx = feature_names.index("Close") if "Close" in feature_names else None
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None

    # ——— trening (pomiar czasu strojenia) ———
    t_tune0 = time.time()
    result_rf, best_rf = train_random_forest(
        X_train, y_train,
        random_state=42, cv_splits=cv_splits_rf, n_iter=n_iter_rf,
        metric=MODEL_METRIC, cv_gap=20, fast=fast_rf,
        target_transform=target_transform,
        close_idx=close_idx, atr_idx=atr_idx
    )
    t_tune = time.time() - t_tune0
    fits_total = cv_splits_rf * n_iter_rf
    print(f"[RF] Czas strojenia: {_fmt(t_tune)}  |  CV: {cv_splits_rf}-fold  |  kandydaci: {n_iter_rf}  "
          f"|  ~{fits_total} fitów + refit  |  fast={fast_rf}")

    assert getattr(best_rf, "n_features_in_", None) == X_train.shape[1], "Random Forest: n_features_in_ mismatch"

    # ——— predykcja + logi ———
    y_pred_rf = predict_random_forest(best_rf, X_test)
    res = evaluate_and_log(result_rf["model"], y_test, y_pred_rf, result_rf.get("Best params", "-"))
    results.append(res)

    # ——— wykresy ———
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_label = f"Random Forest – {preset} ({'residual/delta' if target_transform else 'level'})"
    plot_predictions(y_test, y_pred_rf, dates_test,
        model_name=model_label,
        save_path=os.path.join(RESULTS_DIR, f"random_forest_plot_{preset}.png"))

    perm_dir = os.path.join(RESULTS_DIR_IMPORTANCE, preset); os.makedirs(perm_dir, exist_ok=True)
    analyze_permutation_only(
        model=best_rf, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=perm_dir, model_name=f"Random Forest – {preset}",
        use_residual=(target_transform is not None),
        close_idx=close_idx, atr_idx=atr_idx
    )

# =============================================================================
# 3) XGBOOST – RandomizedSearch + EarlyStopping (final fit na train-val)
# =============================================================================


# =============================================================================
if RUN_XGB_RS_ES:
    print_model_header("[XGBoost] Strojenie (no-ES) → Retrain z EarlyStopping (TSCV) + ewaluacja")
    t0 = time.time()

    # ——— ustawienia ———
    preset = "ta_research_set_XGB"  # ohlc, ta_research_set, mixed
    target_transform   = "delta" 
    cv_splits_xgb      = 5
    n_iter_xgb         = 50
    val_fraction_xgb   = 0.2
    es_rounds_xgb      = 100
    fast_xgb           = False

    # ——— dane ———
    X_train, X_test, y_train, y_test, dates_test, feature_names_from_prepare, _ = prepare_data(
        dataset_type=preset, test_size=0.2, scale_X=False, target_mode="close_next"
    )
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else list(feature_names_from_prepare)
    assert X_train.shape[1] == len(feature_names)
    assert X_test.shape[1]  == len(feature_names)

    close_idx = feature_names.index("Close") if "Close" in feature_names else None
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None

    # ——— tuning + retrain z ES (pomiar czasu) ———
    t_tune0 = time.time()
    result_xgb, best_xgb = train_xgboost(
        X_train, y_train,
        cv_splits=cv_splits_xgb, n_iter=n_iter_xgb,
        val_fraction=val_fraction_xgb, early_stopping_rounds=es_rounds_xgb,
        metric=MODEL_METRIC, cv_gap=20, fast=fast_xgb,
        close_idx=close_idx, atr_idx=atr_idx, target_transform=target_transform
    )
    t_tune = time.time() - t_tune0
    fits_total = cv_splits_xgb * n_iter_xgb
    print(f"[XGB] Czas strojenia+ES: {_fmt(t_tune)}  |  CV: {cv_splits_xgb}-fold  |  kandydaci: {n_iter_xgb} "
          f"|  ~{fits_total} fitów + retrain(ES={es_rounds_xgb})  |  fast={fast_xgb}")

    n_in = getattr(best_xgb, "n_features_in_", None)
    if n_in is not None:
        assert n_in == X_train.shape[1], "XGBoost: n_features_in_ mismatch"

    # ——— predykcja + logi ———
    y_pred_xgb = predict_xgboost(best_xgb, X_test)
    res = evaluate_and_log(result_xgb["model"], y_test, y_pred_xgb, result_xgb.get("Best params", "-"))
    results.append(res)

    # ——— wykresy ———
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_label = f"XGBoost – {preset} ({'residual/delta' if target_transform else 'level'})"
    plot_predictions(y_test, y_pred_xgb, dates_test,
        model_name=model_label,
        save_path=os.path.join(RESULTS_DIR, f"xgboost_plot_{preset}.png"))

    perm_dir = os.path.join(RESULTS_DIR_IMPORTANCE, preset); os.makedirs(perm_dir, exist_ok=True)
    analyze_permutation_only(
        model=best_xgb, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=perm_dir, model_name=f"XGBoost – {preset}",
        use_residual=(target_transform is not None),
        close_idx=close_idx, atr_idx=atr_idx if target_transform == "delta_over_atr" else None
    )

# =============================================================================
# 4) MLP (Grid/RandomizedSearchCV w pipeline + scaling)
# =============================================================================
if RUN_MLP:
    print_model_header("[MLP] Strojenie + ewaluacja")
    t0 = time.time()

    # ——— ustawienia ———
    preset = "mixed"   # ohlc, ta_research_set, mixed
    cv_splits_mlp = 5
    n_iter_mlp    = 30
    fast_mlp      = False

    # — dane — 
    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        dataset_type=preset, test_size=0.2, scale_X=False, target_mode="close_next"
    )

    # ——— tuning (pomiar czasu) ———
    t_tune0 = time.time()
    result_mlp, best_mlp = train_mlp(
        X_train, y_train,
        random_state=42, cv_splits=cv_splits_mlp, n_iter=n_iter_mlp,
        fast=fast_mlp, cv_gap=20
    )
    t_tune = time.time() - t_tune0
    fits_total = cv_splits_mlp * n_iter_mlp
    print(f"[MLP] Czas strojenia: {_fmt(t_tune)}  |  CV: {cv_splits_mlp}-fold  |  kandydaci: {n_iter_mlp}  "
          f"|  ~{fits_total} fitów + refit  |  fast={fast_mlp}")

    # ——— predykcja + logi ———
    y_pred_mlp = predict_mlp(best_mlp, X_test)
    res = evaluate_and_log(result_mlp["model"], y_test, y_pred_mlp, result_mlp.get("Best params", "-"))
    results.append(res)

    plot_predictions(y_test, y_pred_mlp, dates_test,
        model_name="MLPRegressor", save_path=os.path.join(RESULTS_DIR, "mlp_plot.png"))

    analyze_permutation_only(
        model=best_mlp, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=RESULTS_DIR_IMPORTANCE, model_name="MLP", use_residual=False
    )

# =============================================================================
# 5) LSTM (sekwencje)
# =============================================================================
if RUN_LSTM:
    print_model_header("[LSTM] Trening + ewaluacja")

    # ——— ustawienia ———
    preset           = "ta_research_set_LSTM"          # 'ohlc' | 'ta_research_set' | 'mixed'
    target_mode      = "close_next"
    target_transform = "delta"          # NEW: 'level' lub 'delta'
    window_size      = 30
    test_size        = 0.2
    scale_X_flag     = True

    fast_lstm        = False
    units_1_val      = 128
    units_2_val      = 32
    dropout_val      = 0.2
    lr_val           = 3e-4
    es_patience_val  = 16
    rlrop_pat_val    = 6
    epochs_val       = 120
    batch_size_val   = 32

    import os
    RESULTS_DIR_LSTM = os.path.join(RESULTS_DIR, "lstm")
    os.makedirs(RESULTS_DIR_LSTM, exist_ok=True)

    # ——— 1) Dane sekwencyjne ———
    (X_tr_seq, X_te_seq, y_tr_seq, y_te_seq, dates_te_seq,
     scaler_y, feature_names_seq, close_last_test) = create_lstm_data(
        file_path="data/gold_data.csv",
        dataset_type=preset,
        target_mode=target_mode,
        window_size=window_size,
        test_size=test_size,
        scale_X=scale_X_flag,
        target_transform=target_transform,   # NEW
    )
    print(f"[create_lstm_data] Preset='{preset}', cech={X_tr_seq.shape[2]}, okno={window_size}, "
          f"target='{target_mode}', transform='{target_transform}', scale_X={scale_X_flag}")

    # ——— 2) Trening ———
    t0 = time.time()
    model_lstm, history_lstm, _ = train_lstm(
        X_tr_seq, y_tr_seq,
        val_fraction=0.20,
        epochs=epochs_val,
        batch_size=batch_size_val,
        units_1=units_1_val,
        units_2=units_2_val,
        dropout_rate=dropout_val,
        lr=lr_val,
        es_patience=es_patience_val,
        rlrop_patience=rlrop_pat_val,
        verbose=2,
        fast=fast_lstm
    )
    plot_keras_curves(history_lstm, title_prefix=f"LSTM – {preset} ({target_transform})", save_dir=RESULTS_DIR_LSTM)

    # ——— 3) Predykcje — w przestrzeni celu → inverse → ewentualna rekonstrukcja poziomu ———
    y_pred_scaled = predict_lstm(model_lstm, X_te_seq)                               # (skala modelu)
    y_pred_raw    = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel() # (jednostki celu)

    y_true_raw    = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()

    if target_transform == "delta":
        # pred/true to Δ; do poziomu trzeba dodać ostatni Close z okna
        y_pred_level = y_pred_raw + close_last_test
        y_true_level = y_true_raw + close_last_test
    else:
        # pred/true to poziom
        y_pred_level = y_pred_raw
        y_true_level = y_true_raw

    # ——— 4) Ewaluacja + wykres ———
    params_str = (
        f"LSTM(units=({units_1_val},{units_2_val}), dropout={dropout_val}, "
        f"loss=mae, Adam lr={lr_val}, batch={batch_size_val}, "
        f"ES(pat={es_patience_val})+RLROP(pat={rlrop_pat_val}); "
        f"preset={preset}, window={window_size}, tform={target_transform})"
    )
    res = evaluate_and_log("LSTM", y_true_level, y_pred_level, params_str)
    results.append(res)

    plot_predictions(
        y_true_level, y_pred_level, dates_te_seq,
        model_name=f"LSTM – {preset} ({target_transform})",
        save_path=os.path.join(RESULTS_DIR_LSTM, f"lstm_plot_{preset}_{target_transform}.png")
    )

    print(f"[LSTM] Czas: {_fmt(time.time() - t0)} | preset={preset} | okno={window_size} | "
          f"transform={target_transform} | "
          f"MAE={res['MAE mean']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R^2']:.5f}")

    # ——— 5) Permutation importance — y MUSI być w skali modelu (tu: y_te_seq) ———
    analyze_lstm_permutation_only(
        model=model_lstm,
        X_test_seq=X_te_seq,
        y_test_scaled=y_te_seq,            # ta sama skala co wyjście modelu (level lub delta)
        feature_names=feature_names_seq,
        results_dir=RESULTS_DIR_LSTM,
        model_name=f"LSTM – {preset} ({target_transform})",
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

