
from typing import Tuple, Union, Dict, Any, Optional
import inspect
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor


# ─────────────────────────────────────────────────────────────────────────────
# Scoring + refit (kompatybilne z Twoim mainem)
# ─────────────────────────────────────────────────────────────────────────────
def _scoring_and_refit(metric: str) -> Tuple[Union[str, Dict[str, Any]], str]:
    rmse = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                       greater_is_better=False)
    if metric == "mae":
        return "neg_mean_absolute_error", "neg_mean_absolute_error"
    if metric == "rmse":
        return {"rmse": rmse}, "rmse"
    if metric == "r2":
        return "r2", "r2"
    # multi: logujemy MAE+RMSE+R2, refit po MAE
    return {
        "neg_mean_absolute_error": "neg_mean_absolute_error",
        "r2": "r2",
        "rmse": rmse
    }, "neg_mean_absolute_error"


# ─────────────────────────────────────────────────────────────────────────────
# Transformacja celu (jak w DT/RF)
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
# Krok 1 – tuning bez ES (na tej samej przestrzeni celu)
# ─────────────────────────────────────────────────────────────────────────────
def _tune_xgb_without_es(
    X_train, y_train,
    cv_splits: int,
    cv_gap: int,
    n_iter: int,
    metric: str,
    random_state: int,
    fast: bool,
    use_residual_target: bool,          # zachowane dla wstecznej zgodności
    close_idx: Optional[int],
    target_transform: str,
    atr_idx: Optional[int]
):
    # BEZ early_stopping_rounds – ES dopiero w retrenie
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=800 if fast else 1200,
        learning_rate=0.08 if fast else 0.05,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )

    if fast:
        param_distributions = {
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.6, 0.8],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1, 5],
            "min_child_weight": [1, 3],
            "gamma": [0, 1],
        }
        cv_splits = max(3, min(cv_splits, 3))
        n_iter    = min(n_iter, 20)
    else:
        param_distributions = {
            "max_depth": [3, 5, 7, 10, 15],
            "learning_rate": [0.03, 0.05, 0.08],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.9],
            "reg_alpha": [0, 0.1, 1, 10],
            "reg_lambda": [1, 5, 10],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 1, 3],
        }

    scoring, refit_metric = _scoring_and_refit(metric)
    try:
        tscv = TimeSeriesSplit(n_splits=cv_splits, gap=cv_gap)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_metric,
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise"
    )

    # Tuning na tej samej transformacji celu co późniejszy retrain
    y_train_tune = _make_target(y_train, X_train, target_transform, close_idx, atr_idx)
    search.fit(X_train, y_train_tune)

    cvres = search.cv_results_
    def _safe(name): return float(cvres[name][search.best_index_]) if name in cvres else None
    result = {
        "model": "XGBoost (RandomizedSearch, TSCV, no-ES)",
        "CV MAE": -_safe("mean_test_neg_mean_absolute_error") if "mean_test_neg_mean_absolute_error" in cvres else None,
        "CV RMSE": -_safe("mean_test_rmse") if "mean_test_rmse" in cvres else None,
        "CV R2": _safe("mean_test_r2") if "mean_test_r2" in cvres else None,
        "Best params": search.best_params_,
    }
    return result, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# Krok 2 – retrain z ES (na tym samym celu)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_xgb_with_es(
    X_train, y_train,
    best_params: Dict[str, Any],
    val_fraction: float,
    early_stopping_rounds: int,
    metric: str,
    random_state: int,
    target_transform: str,
    close_idx: Optional[int],
    atr_idx: Optional[int]
):
    if not (0 < val_fraction < 0.5):
        raise ValueError("val_fraction musi być w (0, 0.5), np. 0.2")

    n = len(X_train)
    split_idx = int(n * (1 - val_fraction))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Za mało danych do wydzielenia walidacji (ES).")

    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr_raw, y_val_raw = y_train[:split_idx], y_train[split_idx:]

    # Uczenie na tej samej transformacji co tuning
    y_tr  = _make_target(y_tr_raw,  X_tr,  target_transform, close_idx, atr_idx)
    y_val = _make_target(y_val_raw, X_val, target_transform, close_idx, atr_idx)

    est = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=max(1200, best_params.get("n_estimators", 1200)),
        early_stopping_rounds=int(early_stopping_rounds),
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **best_params
    )

    fit_params: Dict[str, Any] = {"eval_set": [(X_val, y_val)]}
    fit_sig = set(inspect.signature(est.fit).parameters.keys())
    if "eval_metric" in fit_sig:
        fit_params["eval_metric"] = "rmse" if metric in ("rmse", "r2", "multi") else "mae"
    if "verbose" in fit_sig:
        fit_params["verbose"] = False

    est.fit(X_tr, y_tr, **fit_params)

    # meta do rekonstrukcji poziomu
    setattr(est, "_target_transform", target_transform)
    setattr(est, "_close_idx", close_idx)
    setattr(est, "_atr_idx", atr_idx)

    return est


# ─────────────────────────────────────────────────────────────────────────────
# Publiczny wrapper
# ─────────────────────────────────────────────────────────────────────────────
def train_xgboost(
    X_train, y_train,
    cv_splits: int = 5,
    random_state: int = 42,
    n_iter: int = 80,
    metric: str = "multi",
    cv_gap: int = 0,
    val_fraction: Optional[float] = 0.2,
    early_stopping_rounds: int = 200,
    fast: bool = False,
    use_residual_target: bool = True,      # dla zgodności ze starym wywołaniem (nieużywane logicznie)
    close_idx: Optional[int] = None,
    target_transform: str = "delta",       # "level" | "delta" | "delta_over_atr" | "return"
    atr_idx: Optional[int] = None
):
    # walidacja transformacji
    if target_transform in ("delta", "return", "delta_over_atr") and close_idx is None:
        raise ValueError(f"target_transform='{target_transform}' wymaga close_idx.")
    if target_transform == "delta_over_atr" and atr_idx is None:
        raise ValueError("target_transform='delta_over_atr' wymaga atr_idx (ATR w cechach).")

    # 1) tuning bez ES
    tune_res, best_params = _tune_xgb_without_es(
        X_train, y_train,
        cv_splits=cv_splits,
        cv_gap=cv_gap,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        fast=fast,
        use_residual_target=use_residual_target,
        close_idx=close_idx,
        target_transform=target_transform,
        atr_idx=atr_idx
    )

    # 2) retrain z ES
    if val_fraction is None:
        raise ValueError("val_fraction nie może być None przy trenowaniu z ES.")
    best_est = _fit_xgb_with_es(
        X_train, y_train,
        best_params=best_params,
        val_fraction=float(val_fraction),
        early_stopping_rounds=early_stopping_rounds,
        metric=metric,
        random_state=random_state,
        target_transform=target_transform,
        close_idx=close_idx,
        atr_idx=atr_idx
    )

    result = {
        "model": "XGBoost (Tuning no-ES → Retrain with ES, TSCV)",
        "CV MAE": tune_res.get("CV MAE"),
        "CV RMSE": tune_res.get("CV RMSE"),
        "CV R2": tune_res.get("CV R2"),
        "Best params": tune_res.get("Best params"),
        "Target transform": target_transform,
    }
    return result, best_est


# ─────────────────────────────────────────────────────────────────────────────
# Predykcja + rekonstrukcja poziomu (jak w DT/RF)
# ─────────────────────────────────────────────────────────────────────────────
def predict_xgboost(model: XGBRegressor, X: Any) -> np.ndarray:
    bi = getattr(model, "best_iteration_", None)
    if isinstance(bi, int) and bi >= 0:
        try:
            yhat = model.predict(X, iteration_range=(0, bi + 1))  # XGB 2.x
        except TypeError:
            yhat = model.predict(X, ntree_limit=bi + 1)            # fallback starsze API
    else:
        yhat = model.predict(X)

    tform    = getattr(model, "_target_transform", "level")
    close_ix = getattr(model, "_close_idx", None)
    atr_ix   = getattr(model, "_atr_idx", None)

    if tform == "level":
        return yhat

    if close_ix is None:
        raise RuntimeError("Brak _close_idx do rekonstrukcji poziomu.")

    close_t = X[:, close_ix]

    if tform == "delta":
        return close_t + yhat

    if tform == "delta_over_atr":
        if atr_ix is None:
            raise RuntimeError("Brak _atr_idx do rekonstrukcji poziomu (delta_over_atr).")
        atr_t = X[:, atr_ix]
        return close_t + yhat * atr_t

    if tform == "return":
        return close_t * (1.0 + yhat)

    return yhat