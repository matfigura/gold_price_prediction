from __future__ import annotations
import os
from joblib import load

from src.feature_viz import (
    analyze_model_features,
    XGBLevelWrapper,
)
from src.data_preprocesing import prepare_data

def main():
    RESULTS_DIR = "results"

    # 1) Załaduj model – tu przykład z DT; dla XGB podmień nazwę pliku
    model = load(os.path.join(RESULTS_DIR, "best_dt.joblib"))

    # 2) Zbuduj dane jak zwykle
    X_train, X_test, y_train, y_test, dates_test, feature_names, _ = prepare_data(
        file_path="data/gold_data.csv",
        dataset_type="ta_lags_2",
        test_size=0.2,
        scale_X=None,
        target_mode="close_next",
    )

    # 3) Indeksy (potrzebne do residualizacji)
    close_idx = feature_names.index("Close")
    atr_idx   = feature_names.index("atr_14") if "atr_14" in feature_names else None

    # 4a) DT/RF uczone na delcie → analiza w przestrzeni residualnej:
    analyze_model_features(
        model=model,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        results_dir=RESULTS_DIR,
        model_name="Decision Tree",
        use_residual=True,             # ← transformujemy y do Δ/Δ/ATR
        close_idx=close_idx,
        atr_idx=atr_idx,
    )

    # 4b) XGB – wariant „na poziomie” (owinięty wrapperem):
    xgb = load(os.path.join(RESULTS_DIR, "best_xgb.joblib"))
    analyze_model_features(
         model=XGBLevelWrapper(xgb),   # ← proxy zwraca poziom
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test,
         feature_names=feature_names,
         results_dir=RESULTS_DIR,
         model_name="XGBoost",
         use_residual=False,           # ← bo wrapper już zwraca poziom
     )

    print("[analyze_features] Wykresy zapisane w:", RESULTS_DIR)

if __name__ == "__main__":
    main()