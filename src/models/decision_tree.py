from typing import Tuple, Union, Dict, Optional, Any, List
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import loguniform


def _scoring_and_refit(metric: str, override_refit: Optional[str] = None) -> Tuple[Union[str, Dict[str, object]], str]:
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


def _tune_dt(
    X_train, y_train,
    cv_splits: int,
    cv_gap: int,
    n_iter: int,
    metric: str,
    random_state: int,
    refit_metric: Optional[str],
    fast: bool,
    target_transform: str,
    close_idx: Optional[int],
):
    base = DecisionTreeRegressor(random_state=random_state)

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
            "min_samples_leaf": [2, 4, 8, 12, 16, 24],
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

    y_tune = _make_target(y_train, X_train, target_transform, close_idx)
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


def train_decision_tree(
    X_train, y_train,
    cv_splits: int = 5,
    random_state: int = 42,
    n_iter: int = 80,
    metric: str = "multi",
    refit_metric: Optional[str] = None,
    cv_gap: int = 0,
    fast: bool = False,
    target_transform: str = "return",
    close_idx: Optional[int] = None,
    feature_scaler: Optional[Any] = None,
    feature_names: Optional[List[str]] = None
):
    if target_transform not in ("level", "return"):
        raise ValueError("target_transform musi być 'level' albo 'return'.")
    if target_transform == "return" and close_idx is None:
        raise ValueError("target_transform='return' wymaga close_idx.")

    tune_res, best_params = _tune_dt(
        X_train, y_train,
        cv_splits=cv_splits,
        cv_gap=cv_gap,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        refit_metric=refit_metric,
        fast=fast,
        target_transform=target_transform,
        close_idx=close_idx,
    )

    best_dt = DecisionTreeRegressor(random_state=random_state, **best_params)
    y_final = _make_target(y_train, X_train, target_transform, close_idx)
    best_dt.fit(X_train, y_final)

    setattr(best_dt, "_target_transform", target_transform)
    setattr(best_dt, "_close_idx", close_idx)
    setattr(best_dt, "_scaler", feature_scaler)
    setattr(best_dt, "_feature_names", feature_names)

    result = {
        "model": "Decision Tree (RandomizedSearch, TSCV)",
        "CV MAE": tune_res.get("CV MAE"),
        "CV RMSE": tune_res.get("CV RMSE"),
        "CV R2": tune_res.get("CV R2"),
        "Best params": tune_res.get("Best params"),
        "Target transform": target_transform,
    }
    return result, best_dt


def predict_decision_tree(model: DecisionTreeRegressor, X: Any) -> np.ndarray:
    yhat_base = model.predict(X)

    tform    = getattr(model, "_target_transform", "level")
    close_ix = getattr(model, "_close_idx", None)
    scaler   = getattr(model, "_scaler", None)

    if tform == "level" or tform is None:
        return yhat_base

    if tform != "return":
        raise RuntimeError("Nieobsługiwany tryb predykcji (oczekiwano 'level' lub 'return').")

    if close_ix is None:
        raise RuntimeError("Brak _close_idx do rekonstrukcji poziomu.")

    close_t = X[:, close_ix]
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        close_t = close_t * scaler.scale_[close_ix] + scaler.mean_[close_ix]

    return close_t * (1.0 + yhat_base)
