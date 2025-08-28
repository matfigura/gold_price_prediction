from typing import Tuple, Union, Dict, Optional, Any
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import loguniform

# ─────────────────────────────────────────────────────────────────────────────
# Scoring + refit (kompatybilne z Twoim mainem)
# ─────────────────────────────────────────────────────────────────────────────
def _scoring_and_refit(metric: str, override_refit: Optional[str] = None) -> Tuple[Union[str, Dict[str, object]], str]:
    """Zwraca (scoring, refit_key)."""
    # [ZMIANA] Bezpieczny RMSE – wbudowany lub własny
    try:
        rmse_key = "neg_root_mean_squared_error"
        rmse_scorer = rmse_key
    except Exception:
        rmse_key = "rmse"
        rmse_scorer = make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)), greater_is_better=True)

    if metric == "mae":
        scoring = "neg_mean_absolute_error"
        refit  = "neg_mean_absolute_error"
    elif metric == "rmse":
        scoring = {rmse_key: rmse_scorer}
        refit  = rmse_key
    elif metric == "r2":
        scoring = "r2"
        refit  = "r2"
    else:
        scoring = {
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "r2": "r2",
            rmse_key: rmse_scorer
        }
        refit = "neg_mean_absolute_error"

    if override_refit is not None:
        refit = override_refit
    return scoring, refit


# ─────────────────────────────────────────────────────────────────────────────
# [ZMIANA] Jednolite tworzenie celu do uczenia (jak w XGB)
# ─────────────────────────────────────────────────────────────────────────────
def _make_target(
    y: np.ndarray, X: np.ndarray,
    target_transform: str,
    close_idx: Optional[int],
    atr_idx: Optional[int]
) -> np.ndarray:
    """
    - "level"           : y
    - "delta"           : y - Close_t
    - "delta_over_atr"  : (y - Close_t) / ATR_t
    - "return"          : (y - Close_t) / Close_t
    """
    if target_transform == "level":
        return y

    if close_idx is None:
        raise ValueError(f"target_transform='{target_transform}' wymaga close_idx (kolumna 'Close' w X).")

    close_t = X[:, close_idx]

    if target_transform == "delta":
        return y - close_t

    if target_transform == "delta_over_atr":
        if atr_idx is None:
            raise ValueError("target_transform='delta_over_atr' wymaga atr_idx (kolumna 'atr_14' w X).")
        atr_t = X[:, atr_idx]
        return (y - close_t) / (atr_t + 1e-8)

    if target_transform == "return":
        return (y - close_t) / (close_t + 1e-8)

    raise ValueError("target_transform ∈ {'level','delta','delta_over_atr','return'}")


# ─────────────────────────────────────────────────────────────────────────────
# [ZMIANA] Tuning DT z TimeSeriesSplit + gap; tryb fast
# ─────────────────────────────────────────────────────────────────────────────
def _tune_dt(
    X_train, y_train,
    cv_splits: int,
    cv_gap: int,
    n_iter: int,
    metric: str,
    random_state: int,
    refit_metric: Optional[str],
    fast: bool,
    target_transform: str,           # [ZMIANA]
    close_idx: Optional[int],        # [ZMIANA]
    atr_idx: Optional[int]           # [ZMIANA]
):
    base = DecisionTreeRegressor(random_state=random_state)

    # [ZMIANA] param grid – mocniej anty-overfit; w fast ciaśniejsza siatka
    if fast:
        param_distributions = {
            "max_depth": [4, 6, 8, 10],
            "min_samples_split": [10, 20, 40],
            "min_samples_leaf": [8, 12, 16, 24],
            "max_features": ["sqrt", "log2", 0.6, 0.8, None],
            "ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3],
            "min_impurity_decrease": [0.0, 1e-7, 1e-6, 1e-5],
            "splitter": ["best", "random"],
        }
        cv_splits = max(3, min(cv_splits, 3))
        n_iter    = min(n_iter, 30)
    else:
        param_distributions = {
            "max_depth": [3, 5, 7, 9, 12, 15],
            "min_samples_split": [2, 5, 10, 20, 40, 80],
            "min_samples_leaf": [2, 4, 8, 12, 16, 24,],
            "max_features": [None, "sqrt", "log2", 0.5, 0.7],
            "ccp_alpha": [0.0, 1e-6, 1e-5, 1e-4, 1e-2],
            "min_impurity_decrease": loguniform(1e-8, 1e-2),
            "splitter": ["best"],
        }

    scoring, refit_key = _scoring_and_refit(metric, override_refit=refit_metric)

    try:
        tscv = TimeSeriesSplit(n_splits=cv_splits, gap=cv_gap)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_key,
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    # [ZMIANA] tuning na tej samej przestrzeni celu, co finalny fit
    y_tune = _make_target(y_train, X_train, target_transform, close_idx, atr_idx)
    search.fit(X_train, y_tune)

    cvres = search.cv_results_

    def _safe(name): 
        return float(cvres[name][search.best_index_]) if name in cvres else None

    rmse_key = "mean_test_neg_root_mean_squared_error" if "mean_test_neg_root_mean_squared_error" in cvres else \
               ("mean_test_rmse" if "mean_test_rmse" in cvres else None)

    result = {
        "model": "Decision Tree (RandomizedSearch, TSCV)",
        "CV MAE": -_safe("mean_test_neg_mean_absolute_error") if "mean_test_neg_mean_absolute_error" in cvres else None,
        "CV RMSE": (-_safe(rmse_key)) if rmse_key else None,
        "CV R2": _safe("mean_test_r2") if "mean_test_r2" in cvres else None,
        "Best params": search.best_params_,
    }
    return result, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# [ZMIANA] Główny wrapper: tuning → finalny fit na całym train
# ─────────────────────────────────────────────────────────────────────────────
def train_decision_tree(
    X_train, y_train,
    cv_splits: int = 5,
    random_state: int = 42,
    n_iter: int = 80,
    metric: str = "multi",
    refit_metric: Optional[str] = None,
    cv_gap: int = 0,
    fast: bool = False,                          # [ZMIANA]
    target_transform: str = "delta",             # [ZMIANA] "level" | "delta" | "delta_over_atr" | "return"
    close_idx: Optional[int] = None,             # [ZMIANA]
    atr_idx: Optional[int] = None                # [ZMIANA]
):
    # [ZMIANA] walidacja parametrów
    if target_transform in ("delta", "return", "delta_over_atr") and close_idx is None:
        raise ValueError(f"target_transform='{target_transform}' wymaga close_idx.")
    if target_transform == "delta_over_atr" and atr_idx is None:
        raise ValueError("target_transform='delta_over_atr' wymaga atr_idx (ATR w cechach).")

    # 1) tuning na tej samej przestrzeni celu
    tune_res, best_params = _tune_dt(
        X_train, y_train,
        cv_splits=cv_splits,
        cv_gap=cv_gap,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        refit_metric=refit_metric,
        fast=fast,
        target_transform=target_transform,  # [ZMIANA]
        close_idx=close_idx,                # [ZMIANA]
        atr_idx=atr_idx                     # [ZMIANA]
    )

    # 2) finalny fit na całym X_train (bez CV)
    best_dt = DecisionTreeRegressor(random_state=random_state, **best_params)
    y_final = _make_target(y_train, X_train, target_transform, close_idx, atr_idx)
    best_dt.fit(X_train, y_final)

    # [ZMIANA] zapisz meta, by helper mógł zrekonstruować poziom
    setattr(best_dt, "_target_transform", target_transform)
    setattr(best_dt, "_close_idx", close_idx)
    setattr(best_dt, "_atr_idx", atr_idx)

    # scalony raport
    result = {
        "model": "Decision Tree (RandomizedSearch, TSCV)",
        "CV MAE": tune_res.get("CV MAE"),
        "CV RMSE": tune_res.get("CV RMSE"),
        "CV R2": tune_res.get("CV R2"),
        "Best params": tune_res.get("Best params"),
        "Target transform": target_transform,   # [ZMIANA] informacyjnie
    }
    return result, best_dt


# ─────────────────────────────────────────────────────────────────────────────
# [ZMIANA] Predykcja: uwzględnia transformację celu, rekonstruuje poziom
# ─────────────────────────────────────────────────────────────────────────────
def predict_decision_tree(model: DecisionTreeRegressor, X: Any) -> np.ndarray:
    yhat_base = model.predict(X)

    tform    = getattr(model, "_target_transform", "level")
    close_ix = getattr(model, "_close_idx", None)
    atr_ix   = getattr(model, "_atr_idx", None)

    if tform == "level":
        return yhat_base

    if close_ix is None:
        raise RuntimeError("Brak _close_idx do rekonstrukcji poziomu (uczenie residualne).")

    close_t = X[:, close_ix]

    if tform == "delta":
        return close_t + yhat_base

    if tform == "delta_over_atr":
        if atr_ix is None:
            raise RuntimeError("Brak _atr_idx do rekonstrukcji (delta_over_atr).")
        atr_t = X[:, atr_ix]
        return close_t + yhat_base * atr_t

    if tform == "return":
        return close_t * (1.0 + yhat_base)

    return yhat_base