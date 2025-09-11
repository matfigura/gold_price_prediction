import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import prepare_data
from src.data_lstm import create_lstm_data

from src.models.decision_tree import train_decision_tree, predict_decision_tree
from src.models.random_forest import train_random_forest, predict_random_forest
from src.models.mlp import train_mlp, predict_mlp
from src.models.xgboost_model import train_xgboost, predict_xgboost

from src.models.lstm import train_lstm, predict_lstm

from src.viz.plots import plot_predictions, build_collage, PLOTS_DIR
from src.feature_viz import analyze_permutation_only, analyze_lstm_permutation_only

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_DIR_IMPORTANCE = os.path.join(RESULTS_DIR, "importance")
os.makedirs(RESULTS_DIR_IMPORTANCE, exist_ok=True)

MODEL_METRIC = "mae"

RUN_DECISION_TREE  = True
RUN_RANDOM_FOREST  = True
RUN_XGB_RS_ES      = True
RUN_MLP            = True
RUN_LSTM           = True

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
    return "neg_mean_absolute_error"

def _fmt(secs: float) -> str:
    m, s = divmod(int(round(secs)), 60)
    h, m = divmod(m, 60)
    return (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")

results = []

if RUN_DECISION_TREE:
    print_model_header("[Decision Tree] Strojenie + ewaluacja")
    t0 = time.time()

    preset = "ta_research_set"
    target_transform = "return"
    cv_splits_dt = 5
    n_iter_dt    = 120

    X_train, X_test, y_train, y_test, dates_test, feature_names_from_prepare, scaler = prepare_data(
        dataset_type=preset,
        test_size=0.2,
        scale_X=False,
        target_mode="close_next"
    )

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else list(feature_names_from_prepare)
    assert X_train.shape[1] == len(feature_names)
    assert X_test.shape[1]  == len(feature_names)

    close_idx = feature_names.index("Close") if "Close" in feature_names else None

    refit_metric = map_metric_to_dt_refit(MODEL_METRIC)

    t_tune0 = time.time()
    result_dt, best_dt = train_decision_tree(
        X_train, y_train,
        cv_splits=cv_splits_dt, n_iter=n_iter_dt,
        metric=MODEL_METRIC, refit_metric=refit_metric,
        cv_gap=20, fast=False,
        target_transform=target_transform,
        close_idx=close_idx,
        feature_scaler=scaler,
        feature_names=feature_names
    )
    t_tune = time.time() - t_tune0

    fits_total = cv_splits_dt * n_iter_dt
    def _fmt(secs: float) -> str:
        m, s = divmod(int(round(secs)), 60)
        h, m = divmod(m, 60)
        return (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")
    print(f"[DT] Czas strojenia: {_fmt(t_tune)}  |  CV: {cv_splits_dt}-fold  |  kandydaci: {n_iter_dt}  "
          f"|  ~{fits_total} fitów + refit")

    assert getattr(best_dt, "n_features_in_", None) == X_train.shape[1]

    y_pred_dt = predict_decision_tree(best_dt, X_test)
    res = evaluate_and_log(result_dt["model"], y_test, y_pred_dt, result_dt.get("Best params", "-"))
    results.append(res)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_label = f"Decision Tree – {preset} ({target_transform})"
    plot_predictions(
        y_test, y_pred_dt, dates_test,
        model_name=model_label,
        save_path=os.path.join(RESULTS_DIR, f"decision_tree_plot_{preset}.png")
    )

    perm_dir = os.path.join(RESULTS_DIR_IMPORTANCE, preset)
    os.makedirs(perm_dir, exist_ok=True)
    analyze_permutation_only(
        model=best_dt,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        results_dir=perm_dir,
        model_name=f"Decision Tree – {preset}",
        use_residual=True,
        close_idx=close_idx
    )

if RUN_RANDOM_FOREST:
    print_model_header("[Random Forest] Strojenie + ewaluacja")
    t0 = time.time()

    preset = "ta_research_set"
    target_transform = "return"
    cv_splits_rf = 5
    n_iter_rf    = 30
    fast_rf      = False

    X_train, X_test, y_train, y_test, dates_test, feature_names_from_prepare, scaler = prepare_data(
        dataset_type=preset, test_size=0.2, scale_X=False, target_mode="close_next"
    )
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else list(feature_names_from_prepare)
    assert X_train.shape[1] == len(feature_names)
    assert X_test.shape[1]  == len(feature_names)

    close_idx = feature_names.index("Close") if "Close" in feature_names else None

    t_tune0 = time.time()
    result_rf, best_rf = train_random_forest(
        X_train, y_train,
        random_state=42, cv_splits=cv_splits_rf, n_iter=n_iter_rf,
        metric=MODEL_METRIC, cv_gap=20, fast=fast_rf,
        target_transform=target_transform,
        close_idx=close_idx,
        feature_scaler=scaler,
        feature_names=feature_names
    )
    t_tune = time.time() - t_tune0
    fits_total = cv_splits_rf * n_iter_rf
    print(f"[RF] Czas strojenia: {_fmt(t_tune)}  |  CV: {cv_splits_rf}-fold  |  kandydaci: {n_iter_rf}  "
          f"|  ~{fits_total} fitów + refit  |  fast={fast_rf}")

    assert getattr(best_rf, "n_features_in_", None) == X_train.shape[1], "Random Forest: n_features_in_ mismatch"

    y_pred_rf = predict_random_forest(best_rf, X_test)
    res = evaluate_and_log(result_rf["model"], y_test, y_pred_rf, result_rf.get("Best params", "-"))
    results.append(res)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_label = f"Random Forest – {preset} ({target_transform})"
    plot_predictions(
        y_test, y_pred_rf, dates_test,
        model_name=model_label,
        save_path=os.path.join(RESULTS_DIR, f"random_forest_plot_{preset}.png")
    )

    perm_dir = os.path.join(RESULTS_DIR_IMPORTANCE, preset); os.makedirs(perm_dir, exist_ok=True)
    analyze_permutation_only(
        model=best_rf, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=perm_dir, model_name=f"Random Forest – {preset}",
        use_residual=True,
        close_idx=close_idx
    )

if RUN_XGB_RS_ES:
    print_model_header("[XGBoost] Strojenie (no-ES) → Retrain z EarlyStopping (TSCV) + ewaluacja")
    t0 = time.time()

    preset = "ta_research_set"
    target_transform = "return"
    cv_splits_xgb    = 5
    n_iter_xgb       = 50
    val_fraction_xgb = 0.2
    es_rounds_xgb    = 100
    fast_xgb         = False

    X_train, X_test, y_train, y_test, dates_test, feature_names_from_prepare, scaler = prepare_data(
        dataset_type=preset, test_size=0.2, scale_X=False, target_mode="close_next"
    )
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else list(feature_names_from_prepare)
    assert X_train.shape[1] == len(feature_names)
    assert X_test.shape[1]  == len(feature_names)

    close_idx = feature_names.index("Close") if "Close" in feature_names else None

    t_tune0 = time.time()
    result_xgb, best_xgb = train_xgboost(
        X_train, y_train,
        cv_splits=cv_splits_xgb, n_iter=n_iter_xgb,
        val_fraction=val_fraction_xgb, early_stopping_rounds=es_rounds_xgb,
        metric=MODEL_METRIC, cv_gap=20, fast=fast_xgb,
        close_idx=close_idx, target_transform=target_transform,
        feature_scaler=scaler, feature_names=feature_names
    )
    t_tune = time.time() - t_tune0
    fits_total = cv_splits_xgb * n_iter_xgb
    print(f"[XGB] Czas strojenia+ES: {_fmt(t_tune)}  |  CV: {cv_splits_xgb}-fold  |  kandydaci: {n_iter_xgb} "
          f"|  ~{fits_total} fitów + retrain(ES={es_rounds_xgb})  |  fast={fast_xgb}")

    n_in = getattr(best_xgb, "n_features_in_", None)
    if n_in is not None:
        assert n_in == X_train.shape[1], "XGBoost: n_features_in_ mismatch"

    y_pred_xgb = predict_xgboost(best_xgb, X_test)
    res = evaluate_and_log(result_xgb["model"], y_test, y_pred_xgb, result_xgb.get("Best params", "-"))
    results.append(res)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_label = f"XGBoost – {preset} ({target_transform})"
    plot_predictions(
        y_test, y_pred_xgb, dates_test,
        model_name=model_label,
        save_path=os.path.join(RESULTS_DIR, f"xgboost_plot_{preset}.png")
    )

    perm_dir = os.path.join(RESULTS_DIR_IMPORTANCE, preset); os.makedirs(perm_dir, exist_ok=True)
    analyze_permutation_only(
        model=best_xgb, X_test=X_test, y_test=y_test, feature_names=feature_names,
        results_dir=perm_dir, model_name=f"XGBoost – {preset}",
        use_residual=True,
        close_idx=close_idx
    )

if RUN_MLP:
    print_model_header("[MLP] Strojenie + ewaluacja")
    t0 = time.time()

    preset = "ohlc"
    cv_splits_mlp = 5
    n_iter_mlp    = 30
    fast_mlp      = False

    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        dataset_type=preset, test_size=0.2, scale_X=False, target_mode="close_next"
    )

    t_tune0 = time.time()

    n_iter_scan = min(12, n_iter_mlp)
    result_mlp_1, best_mlp_1 = train_mlp(
        X_train, y_train,
        random_state=42,
        cv_splits=cv_splits_mlp,
        n_iter=n_iter_scan,
        fast=False,
        cv_gap=20,
        tag=f"stage=1 | preset={preset} | target=close_next"
    )
    best_size = result_mlp_1["Best params"]["mlp__hidden_layer_sizes"]

    result_mlp_2, best_mlp_2 = train_mlp(
        X_train, y_train,
        random_state=42,
        cv_splits=cv_splits_mlp,
        n_iter=n_iter_mlp,
        fast=False,
        cv_gap=20,
        tag=f"stage=2 | preset={preset} | target=close_next | fix_hls={best_size}",
        fix_hidden_size=best_size
    )

    result_mlp, best_mlp = result_mlp_2, best_mlp_2

    t_tune = time.time() - t_tune0

    fits_stage1 = 3 * n_iter_scan
    fits_stage2 = cv_splits_mlp * n_iter_mlp
    fits_total  = fits_stage1 + fits_stage2

    print(f"[MLP] Czas strojenia: {_fmt(t_tune)}  |  CV: {cv_splits_mlp}-fold  "
          f"|  kandydaci (S1/S2): {n_iter_scan}/{n_iter_mlp}  "
          f"|  ~fitów: S1={fits_stage1} + S2={fits_stage2} = {fits_total}  "
          f"|  fast={fast_mlp}  |  preset={preset}")

    n_in = getattr(best_mlp, "n_features_in_", None)
    if n_in is not None:
        assert n_in == X_train.shape[1], "MLP: n_features_in_ mismatch"

    y_pred_mlp = predict_mlp(best_mlp, X_test)
    res = evaluate_and_log(result_mlp["model"], y_test, y_pred_mlp, result_mlp.get("Best params", "-"))
    results.append(res)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_label = f"MLPRegressor – {preset} (level)"
    plot_predictions(
        y_test, y_pred_mlp, dates_test,
        model_name=model_label,
        save_path=os.path.join(RESULTS_DIR, f"mlp_plot_{preset}.png")
    )

    perm_dir = os.path.join(RESULTS_DIR_IMPORTANCE, preset)
    os.makedirs(perm_dir, exist_ok=True)
    analyze_permutation_only(
        model=best_mlp,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        results_dir=perm_dir,
        model_name=f"MLP – {preset}",
        use_residual=False
    )

if RUN_LSTM:
    print_model_header("[LSTM] Trening + ewaluacja")

    preset           = "ta_research_set_LSTM"
    target_mode      = "close_next"
    target_transform = "return"
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

    import os, time
    RESULTS_DIR_LSTM = os.path.join(RESULTS_DIR, "lstm")
    os.makedirs(RESULTS_DIR_LSTM, exist_ok=True)

    (X_tr_seq, X_te_seq, y_tr_seq, y_te_seq, dates_te_seq,
     scaler_y, feature_names_seq, close_last_test) = create_lstm_data(
        file_path="data/gold_data.csv",
        dataset_type=preset,
        target_mode=target_mode,
        window_size=window_size,
        test_size=test_size,
        scale_X=scale_X_flag,
        target_transform=target_transform,
    )
    print(f"[create_lstm_data] Preset='{preset}', cech={X_tr_seq.shape[2]}, okno={window_size}, "
          f"target='{target_mode}', transform='{target_transform}', scale_X={scale_X_flag}")

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

    y_pred_scaled = predict_lstm(model_lstm, X_te_seq)
    y_pred_raw    = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true_raw    = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()

    if target_transform == "return":
        y_pred_level = close_last_test * (1.0 + y_pred_raw)
        y_true_level = close_last_test * (1.0 + y_true_raw)
    else:
        y_pred_level = y_pred_raw
        y_true_level = y_true_raw

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

    analyze_lstm_permutation_only(
        model=model_lstm,
        X_test_seq=X_te_seq,
        y_test_scaled=y_te_seq,
        feature_names=feature_names_seq,
        results_dir=RESULTS_DIR_LSTM,
        model_name=f"LSTM – {preset} ({target_transform})",
        n_repeats=10,
    )

pd.set_option("display.float_format", "{:.5f}".format)
results_df = pd.DataFrame(results)

results_df.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False, float_format="%.5f")

print("\n== PODSUMOWANIE ==")
print(results_df.to_string(index=False,
                           formatters={"MAE mean": "{:.3f}".format,
                                       "RMSE": "{:.3f}".format,
                                       "R^2": "{:.5f}".format}))

build_collage(folder=PLOTS_DIR, pattern="*_plot_*.png", cols=3)
