from __future__ import annotations
import os
import numpy as np
from joblib import load

from src.feature_viz import (
    analyze_permutation_only,
    plot_permutation_importance_generic,
    XGBLevelWrapper,
)
from src.data_preprocesing import prepare_data

# (opcjonalnie) Keras do LSTM
try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None


def main():
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Dane
    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        file_path="data/gold_data.csv",
        dataset_type="ta_lags_2",      # lub inny preset
        test_size=0.2,
        scale_X=None,
        target_mode="close_next",
    )
    close_idx = feature_names.index("Close")
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None

    # 2) DT
    try:
        dt = load(os.path.join(RESULTS_DIR, "best_dt.joblib"))
        analyze_permutation_only(
            model=dt, X_test=X_test, y_test=y_test, feature_names=feature_names,
            results_dir=RESULTS_DIR, model_name="Decision Tree",
            use_residual=True, close_idx=close_idx, atr_idx=atr_idx,  # bo uczyliśmy Δ / Δ/ATR
        )
        print("[analyze_features] DT ✓")
    except Exception as e:
        print("[analyze_features] DT – pominięto:", e)

    # 3) RF
    try:
        rf = load(os.path.join(RESULTS_DIR, "best_rf.joblib"))
        # Jeśli RF był uczony na poziomie → use_residual=False; jeśli na delcie → True
        analyze_permutation_only(
            model=rf, X_test=X_test, y_test=y_test, feature_names=feature_names,
            results_dir=RESULTS_DIR, model_name="Random Forest",
            use_residual=False, close_idx=close_idx, atr_idx=atr_idx,
        )
        print("[analyze_features] RF ✓")
    except Exception as e:
        print("[analyze_features] RF – pominięto:", e)

    # 4) XGBoost
    try:
        xgb = load(os.path.join(RESULTS_DIR, "best_xgb.joblib"))
        # Jeśli XGB był uczony na delcie, można:
        #   a) analizować w przestrzeni poziomu → wraper:
        analyze_permutation_only(
            model=XGBLevelWrapper(xgb),
            X_test=X_test, y_test=y_test, feature_names=feature_names,
            results_dir=RESULTS_DIR, model_name="XGBoost",
            use_residual=False, close_idx=close_idx, atr_idx=atr_idx,
        )
        print("[analyze_features] XGB ✓")
    except Exception as e:
        print("[analyze_features] XGB – pominięto:", e)

    # 5) MLP (sklearn)
    try:
        mlp = load(os.path.join(RESULTS_DIR, "best_mlp.joblib"))
        plot_permutation_importance_generic(
            predict_fn=mlp.predict,
            X=X_test, y=y_test, feature_names=feature_names,
            scoring="mae", n_repeats=30, block_size=20, clip_at_zero=True, top_n=20,
            title="MLP – permutation importance (MAE)",
            save_path=os.path.join(RESULTS_DIR, "mlp_perm_mae.png"),
        )
        print("[analyze_features] MLP ✓")
    except Exception as e:
        print("[analyze_features] MLP – pominięto:", e)

    # 6) LSTM (Keras)
    if keras_load_model is not None:
        for fname in ("best_lstm.h5", "best_lstm.keras"):
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                try:
                    lstm = keras_load_model(path)
                    # Uwaga: tu zakładamy, że masz X_test przygotowane 3D (n, T, d).
                    # Jeśli nie – użyj odpowiedniego tensora zamiast X_test.
                    plot_permutation_importance_generic(
                        predict_fn=lambda X: lstm.predict(X, verbose=0).ravel(),
                        X=X_test, y=y_test, feature_names=feature_names,
                        scoring="mae", n_repeats=20, block_size=None, clip_at_zero=True, top_n=20,
                        title="LSTM – permutation importance (MAE)",
                        save_path=os.path.join(RESULTS_DIR, "lstm_perm_mae.png"),
                    )
                    print("[analyze_features] LSTM ✓")
                except Exception as e:
                    print("[analyze_features] LSTM – pominięto:", e)
                break

    print(f"[analyze_features] Wykresy zapisane w: {RESULTS_DIR}")


if __name__ == "__main__":
    main()