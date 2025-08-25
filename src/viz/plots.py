from __future__ import annotations

import os
from typing import Optional, Sequence, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

try:
    # jeśli masz helper w tym miejscu:
    from src.models.xgboost_model import predict_xgboost as _xgb_predict_level
except Exception:
    _xgb_predict_level = None

class XGBLevelWrapper:
    """Proxy-estymator, który zwraca predykcję na poziomie (Close_t + Δ̂ / itp.)."""
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        if _xgb_predict_level is None:
            # awaryjnie: surowa predykcja
            return self.model.predict(X)
        return _xgb_predict_level(self.model, X)


def plot_tree_impurity_importance(
    model: Any,
    feature_names: Sequence[str],
    top_n: int = 20,
    title: str = "Feature importance (impurity-based)",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Działa dla modeli z atrybutem .feature_importances_ (DT/RF/GB/XGB).
    """
    if not hasattr(model, "feature_importances_"):
        print("[plot_tree_impurity_importance] model nie ma .feature_importances_ → pomijam.")
        return

    importances = np.asarray(model.feature_importances_)
    names = np.asarray(feature_names)
    order = np.argsort(importances)[::-1]
    order = order[: min(top_n, len(order))]

    plt.figure(figsize=figsize)
    plt.barh(range(len(order)), importances[order][::-1])
    plt.yticks(range(len(order)), names[order][::-1])
    plt.xlabel("Ważność (impurity)")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    use_residual: bool = False,
    close_idx: Optional[int] = None,
    atr_idx: Optional[int] = None,
    scoring: str = "neg_mean_absolute_error",
    n_repeats: int = 30,
    random_state: int = 42,
    top_n: Optional[int] = None,
    sort_abs: bool = True,
    title: Optional[str] = None,        # ← DODANE
    save_path: Optional[str] = None,
):
    """Permutation importance; jeśli use_residual=True, to liczymy na Δ lub Δ/ATR."""
    # dopasuj cel (poziom vs residual)
    if use_residual:
        if close_idx is None:
            raise ValueError("use_residual=True wymaga close_idx")
        if atr_idx is None:
            y_pi = y - X[:, close_idx]                    # Δ
        else:
            eps = 1e-8
            y_pi = (y - X[:, close_idx]) / (X[:, atr_idx] + eps)  # Δ/ATR
    else:
        y_pi = y

    pi = permutation_importance(
        model, X, y_pi,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    # sklearn zwraca spadek "wyniku". Dla neg_* metryk (np. neg_mae) robi się ujemnie.
    importances = pi.importances_mean
    if scoring.startswith("neg_"):
        importances = -importances     # pokazujemy wzrost właściwej straty (np. MAE) → dodatnie słupki

    # sortowanie i przycięcie
    order = np.argsort(np.abs(importances)) if sort_abs else np.argsort(importances)
    names_sorted = np.array(feature_names)[order]
    vals_sorted = importances[order]
    if top_n is not None and top_n > 0:
        names_sorted = names_sorted[-top_n:]
        vals_sorted  = vals_sorted[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names_sorted, vals_sorted)
    ax.set_title(title or "Permutation importance")
    ax.set_xlabel("Wzrost błędu po permutacji" + (f" (scoring={scoring})" if scoring else ""))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    use_residual: bool = False,
    close_idx: Optional[int] = None,
    atr_idx: Optional[int] = None,   # używane tylko gdy residual = Δ/ATR
    method: str = "spearman",
    title: str = "Korelacje: cechy vs cel",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 10),
    annot: bool = False,
) -> None:
    """
    Rysuje heatmapę korelacji cech + celu (poziom lub „residual” – Δ lub Δ/ATR).
    - use_residual=False  → cel = y
    - use_residual=True   → cel = (y - Close_t) albo (y - Close_t)/ATR_t (jeśli atr_idx podany)
    """
    import pandas as pd

    if use_residual:
        if close_idx is None:
            raise ValueError("use_residual=True wymaga close_idx (pozycja kolumny Close w X).")
        delta = y - X[:, close_idx]
        if atr_idx is not None:
            # Δ/ATR – stabilizacja skali
            atr = X[:, atr_idx]
            eps = 1e-8
            target_vec = delta / (atr + eps)
            target_name = "target_residual_delta_over_atr"
        else:
            target_vec = delta
            target_name = "target_residual_delta"
    else:
        target_vec = y
        target_name = "target_level"

    df = pd.DataFrame(X, columns=list(feature_names))
    df[target_name] = target_vec

    corr = df.corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        annot=annot,
        fmt=".2f" if annot else "",
        cbar_kws={"shrink": 0.8, "label": f"Korelacja ({method})"},
        square=False,
    )
    plt.title(title)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=160)
    plt.close()


def analyze_model_features(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    results_dir: str,
    model_name: str = "Model",
    *,
    use_residual: bool = False,
    close_idx: Optional[int] = None,
    atr_idx: Optional[int] = None,
) -> None:
    """
    Wygodny wrapper: odpal trzy wykresy jednym wywołaniem.
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1) impurity (jeśli model ma feature_importances_)
    plot_tree_impurity_importance(
        model, feature_names,
        top_n=20,
        title=f"{model_name} – impurity feature importance",
        save_path=os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_impurity.png"),
    )

    # 2) permutation (na zbiorze testowym)
    plot_permutation_importance(
        model, X_test, y_test, feature_names,
        n_repeats=30, scoring="neg_mean_absolute_error", top_n=20,
        title=f"{model_name} – permutation importance (MAE)",
        save_path=os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_perm_mae.png"),
    )

    # 3) heatmapa korelacji (na train, zwykle stabilniej)
    plot_corr_heatmap(
        X_train, y_train, feature_names,
        use_residual=use_residual,
        close_idx=close_idx,
        atr_idx=atr_idx,
        method="spearman",
        title=f"Korelacje (train) – {model_name}",
        save_path=os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_corr_train.png"),
    )