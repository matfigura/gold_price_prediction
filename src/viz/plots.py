from __future__ import annotations

import os
import re
from typing import Optional, Sequence, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _align_feature_names(feature_names: Optional[Sequence[str]], n_feat: int) -> np.ndarray:
    """Zapewnia, że mamy dokładnie n_feat nazw: przycina lub dopisuje f{i}."""
    if feature_names is None:
        return np.array([f"f{i}" for i in range(n_feat)], dtype=object)
    names = list(feature_names)
    if len(names) > n_feat:
        names = names[:n_feat]
    elif len(names) < n_feat:
        names += [f"f{i}" for i in range(len(names), n_feat)]
    return np.array(names, dtype=object)


def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip().lower())


# (opcjonalnie) dla XGBoost uczonego na delcie — wrapper, który zwraca poziom
try:
    from src.models.xgboost_model import predict_xgboost as _xgb_predict_level
except Exception:
    _xgb_predict_level = None

class XGBLevelWrapper:
    """Proxy-estymator, który zwraca predykcję na POZIOMIE (Close_t + Δ̂ / itp.)."""
    def __init__(self, model: Any):
        self.model = model
    def predict(self, X: np.ndarray) -> np.ndarray:
        if _xgb_predict_level is None:
            return self.model.predict(X)
        return _xgb_predict_level(self.model, X)


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE – wersja generyczna (predykcja już w skali celu)
# ─────────────────────────────────────────────────────────────────────────────
def permutation_importance_generic(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    scoring: str = "mae",                # "mae" | "rmse"
    n_repeats: int = 20,
    random_state: int = 42,
    block_size: Optional[int] = None,    # dla 2D: permutacja blokowa (szeregi)
) -> np.ndarray:
    """
    Zwraca wektor importances: średni WZROST błędu po permutacji danej cechy.
    - Obsługuje X o kształcie (n, d) oraz (n, T, d) (np. wejście do LSTM).
    - Dla block_size>1 (tylko X 2D) stosuje permutację blokową zamiast pełnego shuffle.
    Zakłada, że predict_fn zwraca predykcję w **tej samej skali co y** (bez rekonstrukcji).
    """
    rng = np.random.default_rng(random_state)

    def _score(y_true, y_pred):
        if scoring == "rmse":
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return float(np.mean(np.abs(y_true - y_pred)))  # MAE

    X = X.values if hasattr(X, "values") else np.asarray(X)
    y = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()

    base_pred = predict_fn(X)
    base_err  = _score(y, base_pred)

    if X.ndim == 2:
        n, d = X.shape
    elif X.ndim == 3:
        n, T, d = X.shape
    else:
        raise ValueError("X must be 2D or 3D")

    importances = np.zeros(d, dtype=float)

    for j in range(d):
        diffs = []
        for _ in range(n_repeats):
            Xp = X.copy()

            if X.ndim == 2:
                col = Xp[:, j]
                if block_size and block_size > 1:
                    idx = np.arange(n)
                    blocks = [idx[k:k+block_size] for k in range(0, n, block_size)]
                    rng.shuffle(blocks)
                    perm = np.concatenate(blocks)[:n]
                    Xp[:, j] = col[perm]
                else:
                    rng.shuffle(col)
                    Xp[:, j] = col
            else:
                # 3D: mieszamy próbki (batch) całymi sekwencjami cechy j
                col = Xp[:, :, j]
                perm = rng.permutation(n)
                Xp[:, :, j] = col[perm, :]

            err = _score(y, predict_fn(Xp))
            diffs.append(err - base_err)

        importances[j] = float(np.mean(diffs))

    return importances


def plot_permutation_importance_generic(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    scoring: str = "mae",
    n_repeats: int = 20,
    random_state: int = 42,
    block_size: Optional[int] = None,
    clip_at_zero: bool = False,       # True → mikro-ujemności obcinane do 0
    top_n: Optional[int] = None,
    title: str = "Permutation importance",
    save_path: Optional[str] = None,
):
    """
    Rysuje permutation importance dla dowolnego modelu, który ma predict_fn(X)->ŷ
    w **tej samej skali co y**.
    """
    imps = permutation_importance_generic(
        predict_fn, X, y,
        scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, block_size=block_size
    )
    if clip_at_zero:
        imps = np.maximum(imps, 0.0)

    order = np.argsort(imps)
    names_sorted = np.array(feature_names)[order]
    vals_sorted  = imps[order]
    if top_n is not None and top_n > 0:
        names_sorted = names_sorted[-top_n:]
        vals_sorted  = vals_sorted[-top_n:]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(names_sorted, vals_sorted)
    ax.set_xlabel(f"Wzrost błędu po permutacji ({scoring.upper()})")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE – modele uczone na DELCIE (rekonstrukcja do poziomu)
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    use_residual: bool = False,
    close_idx: Optional[int] = None,
    atr_idx: Optional[int] = None,
    scoring: str = "mae",
    n_repeats: int = 30,
    random_state: int = 42,
    block_size: Optional[int] = 20,
    clip_at_zero: bool = True,
    top_n: Optional[int] = 20,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    eval_space: Optional[str] = None,   # "residual" | "level" | None (auto)
    pred_kind: str = "auto",            # "auto" | "residual" | "level"
) -> pd.DataFrame:
    """
    Oblicza permutation importance dla modeli trenowanych na poziomie lub na delcie.
    - eval_space: w jakiej przestrzeni oceniamy ważność (domyślnie: 'residual' jeśli use_residual=True, inaczej 'level').
    - pred_kind: w jakiej przestrzeni zwraca predict(); 'auto' sam wykrywa.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    import pandas as pd

    # --- konwersje i sanity ---
    X = X.values if hasattr(X, "values") else np.asarray(X)
    y = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()
    n, p = X.shape
    feature_names = _align_feature_names(feature_names, p)

    if eval_space is None:
        eval_space = "residual" if use_residual else "level"
    if eval_space not in {"residual", "level"}:
        raise ValueError("eval_space must be 'residual' or 'level'")
    if scoring != "mae":
        raise ValueError("Obsługujemy tylko scoring='mae'")

    rng = np.random.default_rng(random_state)

    # --- pomocnicze przekształcenia między przestrzeniami ---
    def to_level(y_resid: np.ndarray, X_like: np.ndarray) -> np.ndarray:
        if atr_idx is not None:
            return y_resid * X_like[:, atr_idx] + X_like[:, close_idx]
        return y_resid + X_like[:, close_idx]

    def to_residual(y_level: np.ndarray, X_like: np.ndarray) -> np.ndarray:
        if close_idx is None:
            raise ValueError("Konwersja level→residual wymaga close_idx")
        if atr_idx is not None:
            return (y_level - X_like[:, close_idx]) / (X_like[:, atr_idx] + 1e-8)
        return y_level - X_like[:, close_idx]

    mae = lambda a, b: float(np.mean(np.abs(a - b)))

    # docelowy wektor odniesienia (y w przestrzeni ewaluacji)
    if eval_space == "residual":
        if close_idx is None:
            raise ValueError("eval_space='residual' wymaga close_idx")
        y_ref = to_residual(y, X)
    else:
        y_ref = y

    # --- auto-detekcja przestrzeni predykcji ---
    y_hat_raw = model.predict(X)
    if pred_kind == "auto":
        if use_residual:
            # jeżeli model zwraca poziom, będzie miał dużo mniejsze MAE do y (level) niż do y_residual
            err_to_level   = mae(y, y_hat_raw)
            err_to_resid   = mae(to_residual(y, X), y_hat_raw)
            pred_kind = "level" if err_to_level < err_to_resid else "residual"
        else:
            pred_kind = "level"  # gdy nie używamy residuali, zwykle przewidujemy poziom
    elif pred_kind not in {"residual", "level"}:
        raise ValueError("pred_kind must be 'auto' | 'residual' | 'level'")

    # funkcja zwracająca predykcję w PRZESTRZENI EWALUACJI (residual/level)
    def predict_in_eval_space(X_like: np.ndarray) -> np.ndarray:
        y_hat = model.predict(X_like)
        if eval_space == "residual":
            # potrzebujemy predykcji w residual space
            return y_hat if pred_kind == "residual" else to_residual(y_hat, X_like)
        else:
            # potrzebujemy predykcji w level space
            return y_hat if pred_kind == "level" else to_level(y_hat, X_like)

    # baseline
    y_pred0 = predict_in_eval_space(X)
    mae0 = mae(y_ref, y_pred0)

    # permutator (blokowy)
    def permute_vec(v: np.ndarray) -> np.ndarray:
        v = v.copy()
        if block_size is None or block_size <= 1:
            rng.shuffle(v)
            return v
        idx = np.arange(len(v))
        blocks = [idx[i:i+block_size] for i in range(0, len(v), block_size)]
        order = rng.permutation(len(blocks))
        return v[np.concatenate([blocks[k] for k in order])]

    # główna pętla
    inc = np.zeros(p, dtype=float)
    for j in range(p):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = permute_vec(Xp[:, j])
            y_hat_p = predict_in_eval_space(Xp)
            deltas.append(mae(y_ref, y_hat_p) - mae0)
        inc[j] = float(np.mean(deltas))

    if clip_at_zero:
        inc = np.maximum(inc, 0.0)

    order = np.argsort(inc)[::-1]
    feat_sorted = feature_names[order]
    imp_sorted  = inc[order]
    if top_n is not None:
        feat_sorted = feat_sorted[:top_n]
        imp_sorted  = imp_sorted[:top_n]

    # wykres
    plt.figure(figsize=(12, 6))
    plt.barh(feat_sorted[::-1], imp_sorted[::-1])
    plt.xlabel("Wzrost błędu po permutacji (MAE)")
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return pd.DataFrame({"feature": feat_sorted, "importance": imp_sorted})


# „fasada” eksportowana dalej – proste API do jednego wykresu na koniec pipeline’u
def analyze_permutation_only(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    results_dir: str,
    model_name: str,
    *,
    use_residual: bool = False,
    close_idx: Optional[int] = None,
    atr_idx: Optional[int] = None,
):
    os.makedirs(results_dir, exist_ok=True)
    slug = _safe_slug(model_name)
    return plot_model_permutation_importance(
        model=model, X=X_test, y=y_test, feature_names=feature_names,
        use_residual=use_residual, close_idx=close_idx, atr_idx=atr_idx,
        scoring="mae", n_repeats=30, block_size=20, clip_at_zero=True, top_n=20,
        title=f"{model_name} – permutation importance (MAE)",
        save_path=os.path.join(results_dir, f"{slug}_perm_mae.png"),
        eval_space=("residual" if use_residual else "level"),
        pred_kind="auto",
        
    )


# ─────────────────────────────────────────────────────────────────────────────
# LSTM – permutation importance (predykcja i y w tej samej skali)
# ─────────────────────────────────────────────────────────────────────────────
def plot_lstm_permutation_importance(
    model: Any,
    X_seq: np.ndarray,               # (n_seq, window, n_feat)
    y: np.ndarray,                   # UWAGA: w TEJ SAMEJ skali co predykcja modelu
    feature_names: Sequence[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    clip_negatives: bool = True,     # mikro-ujemności obcinane do 0
    title: str = "LSTM – permutation importance (MAE)",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Permutacja po cechach dla LSTM:
    Dla każdej cechy j i każdego kroku czasowego t permutowane są wartości X[:, t, j]
    między sekwencjami (batch), zachowując porządek w czasie w obrębie sekwencji.
    Wartością ważności jest wzrost MAE względem bazowego MAE.
    """
    rng = np.random.default_rng(random_state)

    X_seq = np.asarray(X_seq)
    y = np.asarray(y).ravel()
    n_feat = X_seq.shape[2]
    feature_names = _align_feature_names(feature_names, n_feat)

    # baseline
    y_pred_base = model.predict(X_seq, verbose=0).ravel()
    baseline_mae = float(np.mean(np.abs(y - y_pred_base)))

    importances = np.zeros(n_feat, dtype=float)

    for j in range(n_feat):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_seq.copy()
            for t in range(Xp.shape[1]):
                rng.shuffle(Xp[:, t, j])  # permutacja między sekwencjami
            y_pred_perm = model.predict(Xp, verbose=0).ravel()
            mae_perm = float(np.mean(np.abs(y - y_pred_perm)))
            deltas.append(mae_perm - baseline_mae)
        imp = float(np.mean(deltas))
        importances[j] = max(0.0, imp) if clip_negatives else imp

    order = np.argsort(importances)[::-1]
    names_sorted = feature_names[order]
    vals_sorted  = importances[order]

    # wykres
    plt.figure(figsize=(10, 7))
    plt.barh(names_sorted[::-1], vals_sorted[::-1])
    plt.title(title)
    plt.xlabel("Wzrost MAE po permutacji")
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

    return pd.DataFrame({"feature": names_sorted, "importance": vals_sorted})


def analyze_lstm_permutation_only(
    model: Any,
    X_test_seq: np.ndarray,          # (n_seq, window, n_feat)
    y_test_scaled: np.ndarray,       # ta sama skala co wyjście modelu
    feature_names: Sequence[str],
    results_dir: str,
    model_name: str = "LSTM",
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Wygodny wrapper: jeden call → zapis wykresu permutation importance dla LSTM.
    """
    os.makedirs(results_dir, exist_ok=True)
    slug = _safe_slug(model_name)
    save_path = os.path.join(results_dir, f"{slug}_perm_mae.png")
    return plot_lstm_permutation_importance(
        model=model,
        X_seq=X_test_seq,
        y=y_test_scaled,
        feature_names=feature_names,
        n_repeats=n_repeats,
        random_state=random_state,
        clip_negatives=True,
        title=f"{model_name} – permutation importance (MAE)",
        save_path=save_path,
    )

def plot_keras_curves(history, title_prefix="LSTM", save_dir="results"):
    """Rysuje krzywe uczenia Keras (loss oraz – jeśli dostępne – MAE)."""
    import matplotlib.pyplot as plt
    hist = history.history

    # 1) Loss (u Ciebie loss = MAE, więc to realnie MAE)
    plt.figure(figsize=(8, 4))
    plt.plot(hist.get("loss", []), label="train")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val")
    plt.xlabel("Epoka")
    plt.ylabel("Loss (MAE)")
    plt.title(f"{title_prefix} – loss vs epoka")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_dir}/{title_prefix.lower()}_loss_curve.png", dpi=140)
    plt.close()

    # 2) Opcjonalnie: osobny wykres MAE, jeśli masz metrics=['mae'] w compile()
    mae_key = "mae" if "mae" in hist else ("mean_absolute_error" if "mean_absolute_error" in hist else None)
    if mae_key:
        val_mae_key = "val_mae" if "val_mae" in hist else ("val_mean_absolute_error" if "val_mean_absolute_error" in hist else None)
        plt.figure(figsize=(8, 4))
        plt.plot(hist[mae_key], label="train")
        if val_mae_key:
            plt.plot(hist[val_mae_key], label="val")
        plt.xlabel("Epoka")
        plt.ylabel("MAE")
        plt.title(f"{title_prefix} – MAE vs epoka")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{save_dir}/{title_prefix.lower()}_mae_curve.png", dpi=140)
        plt.close()