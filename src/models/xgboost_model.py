from typing import Tuple, Union, Dict, Any, Optional, List
import inspect
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor


def _scoring_and_refit(metric: str) -> Tuple[Union[str, Dict[str, Any]], str]:
    rmse = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                       greater_is_better=False)
    if metric == "mae":
        return "neg_mean_absolute_error", "neg_mean_absolute_error"
    if metric == "rmse":
        return {"rmse": rmse}, "rmse"
    if metric == "r2":
        return "r2", "r2"

    return {
        "neg_mean_absolute_error": "neg_mean_absolute_error",
        "r2": "r2",
        "rmse": rmse
    }, "neg_mean_absolute_error"


def _make_target(
    y: np.ndarray, X: np.ndarray,
    target_transform: str,
    close_idx: Optional[int],
) -> np.ndarray:
    """
    - "level"  : y
    - "return" : (y - Close_t) / Close_t
    """
    if target_transform == "level" or target_transform is None:
        return y
    if target_transform != "return":
        raise ValueError("target_transform musi być 'level' albo 'return'.")
    if close_idx is None:
        raise ValueError("target_transform='return' wymaga close_idx (kolumna 'Close' w X).")
    close_t = X[:, close_idx]
    return (y - close_t) / (close_t + 1e-8)

def _tune_xgb_without_es(
    X_train, y_train,
    cv_splits: int,
    cv_gap: int,
    n_iter: int,
    metric: str,
    random_state: int,
    fast: bool,
    target_transform: str,
    close_idx: Optional[int],
):
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

    y_tune = _make_target(y_train, X_train, target_transform, close_idx)
    search.fit(X_train, y_tune)

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


def _fit_xgb_with_es(
    X_train, y_train,
    best_params: Dict[str, Any],
    val_fraction: float,
    early_stopping_rounds: int,
    metric: str,
    random_state: int,
    target_transform: str,
    close_idx: Optional[int],
):
    if not (0 < val_fraction < 0.5):
        raise ValueError("val_fraction musi być w (0, 0.5), np. 0.2")

    n = len(X_train)
    split_idx = int(n * (1 - val_fraction))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Za mało danych do wydzielenia walidacji (ES).")

    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr_raw, y_val_raw = y_train[:split_idx], y_train[split_idx:]


    y_tr  = _make_target(y_tr_raw,  X_tr,  target_transform, close_idx)
    y_val = _make_target(y_val_raw, X_val, target_transform, close_idx)

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


    setattr(est, "_target_transform", target_transform)
    setattr(est, "_close_idx", close_idx)


    return est


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
    close_idx: Optional[int] = None,
    target_transform: str = "return",       
    feature_scaler: Optional[Any] = None,
    feature_names: Optional[List[str]] = None
):

    if target_transform not in ("level", "return"):
        raise ValueError("target_transform musi być 'level' albo 'return'.")
    if target_transform == "return" and close_idx is None:
        raise ValueError("target_transform='return' wymaga close_idx.")

 
    tune_res, best_params = _tune_xgb_without_es(
        X_train, y_train,
        cv_splits=cv_splits,
        cv_gap=cv_gap,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        fast=fast,
        target_transform=target_transform,
        close_idx=close_idx,
    )


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
    )

 
    setattr(best_est, "_scaler", feature_scaler)
    setattr(best_est, "_feature_names", feature_names)

    result = {
        "model": "XGBoost (Tuning no-ES → Retrain with ES, TSCV)",
        "CV MAE": tune_res.get("CV MAE"),
        "CV RMSE": tune_res.get("CV RMSE"),
        "CV R2": tune_res.get("CV R2"),
        "Best params": tune_res.get("Best params"),
        "Target transform": target_transform,
    }
    return result, best_est



def predict_xgboost(model: XGBRegressor, X: Any) -> np.ndarray:
    bi = getattr(model, "best_iteration_", None)
    if isinstance(bi, int) and bi >= 0:
        try:
            yhat = model.predict(X, iteration_range=(0, bi + 1))  
        except TypeError:
            yhat = model.predict(X, ntree_limit=bi + 1)            
    else:
        yhat = model.predict(X)

    tform    = getattr(model, "_target_transform", "level")
    close_ix = getattr(model, "_close_idx", None)
    scaler   = getattr(model, "_scaler", None)

    if tform == "level":
        return yhat

    if tform != "return":
        raise RuntimeError("Nieobsługiwany tryb predykcji (oczekiwano 'level' lub 'return').")

    if close_ix is None:
        raise RuntimeError("Brak _close_idx do rekonstrukcji poziomu.")


    close_t = X[:, close_ix]
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        close_t = close_t * scaler.scale_[close_ix] + scaler.mean_[close_ix]

    return close_t * (1.0 + yhat)