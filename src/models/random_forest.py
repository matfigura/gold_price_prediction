from __future__ import annotations

from typing import Tuple, Union, Dict, Optional, Any, List
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error


def _scoring_and_refit(metric: str) -> Tuple[Union[str, Dict[str, object]], str]:
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



def _tune_rf(
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

    y_tune = _make_target(y_train, X_train, target_transform, close_idx)

    base = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False
    )

    if fast:
        param_distributions = {
            "n_estimators": [200, 400],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.6, 0.8],
            "max_samples": [None, 0.7, 0.9],
        }
        cv_splits = max(3, min(cv_splits, 3))
        n_iter    = min(n_iter, 30)
    else:
        param_distributions = {
            "n_estimators": [400, 800, 1200],
            "max_depth": [5, 10, 20, 40, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8, 16],
            "max_features": [None, "sqrt", "log2", 0.5, 0.7],
            "max_samples": [None, 0.7, 0.9],
        }

    scoring, refit = _scoring_and_refit(metric)

    try:
        tscv = TimeSeriesSplit(n_splits=cv_splits, gap=cv_gap)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit,
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    search.fit(X_train, y_tune)

    cvres = search.cv_results_
    def _safe(name): 
        return float(cvres[name][search.best_index_]) if name in cvres else None

    result = {
        "model": "Random Forest (RandomizedSearch, TSCV)",
        "CV MAE": -_safe("mean_test_neg_mean_absolute_error") if "mean_test_neg_mean_absolute_error" in cvres else None,
        "CV RMSE": -_safe("mean_test_rmse") if "mean_test_rmse" in cvres else None,
        "CV R2": _safe("mean_test_r2") if "mean_test_r2" in cvres else None,
        "Best params": search.best_params_,
    }
    return result, search.best_params_



def train_random_forest(
    X_train, y_train,
    random_state: int = 42,
    cv_splits: int = 5,
    n_iter: int = 60,
    metric: str = "multi",
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

 
    tune_res, best_params = _tune_rf(
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


    best_rf = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False,
        **best_params
    )
    y_final = _make_target(y_train, X_train, target_transform, close_idx)
    best_rf.fit(X_train, y_final)


    setattr(best_rf, "_target_transform", target_transform)
    setattr(best_rf, "_close_idx", close_idx)
    setattr(best_rf, "_scaler", feature_scaler)
    setattr(best_rf, "_feature_names", feature_names)


    result = {
        "model": "Random Forest (RandomizedSearch, TSCV)",
        "CV MAE": tune_res.get("CV MAE"),
        "CV RMSE": tune_res.get("CV RMSE"),
        "CV R2": tune_res.get("CV R2"),
        "Best params": tune_res.get("Best params"),
        "Target transform": target_transform,
    }
    return result, best_rf



def predict_random_forest(model: RandomForestRegressor, X: Any) -> np.ndarray:
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