from __future__ import annotations

import os
from typing import Optional, Sequence, Any, Callable

import numpy as np
import matplotlib.pyplot as plt

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


# ─────────────────────────────────────────────────────────────────────────────
# utils
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


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
# PERMUTATION IMPORTANCE – wersja uniwersalna (działa też dla MLP/LSTM)
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
    - Dla block_size>0 (tylko X 2D) stosuje permutację blokową zamiast pełnego shuffle.
    """
    rng = np.random.default_rng(random_state)

    def _score(y_true, y_pred):
        if scoring == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        return np.mean(np.abs(y_true - y_pred))  # MAE (domyślnie)

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
                    # permutacja blokowa: mieszamy całe bloki indeksów (chroni lokalny porządek)
                    idx = np.arange(n)
                    blocks = [idx[k:k+block_size] for k in range(0, n, block_size)]
                    rng.shuffle(blocks)
                    perm = np.concatenate(blocks)[:n]
                    Xp[:, j] = col[perm]
                else:
                    rng.shuffle(col)
                    Xp[:, j] = col
            else:
                # 3D: mieszamy próbki (batch) całymi sekwencjami cechy j → zachowujemy porządek w czasie
                col = Xp[:, :, j]
                perm = rng.permutation(n)
                Xp[:, :, j] = col[perm, :]

            err = _score(y, predict_fn(Xp))
            diffs.append(err - base_err)   # wzrost błędu

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
    Rysuje permutation importance dla dowolnego modelu, który ma predict_fn(X)->ŷ.
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


def plot_model_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    # jeśli model był uczony na delcie → True (y zostanie przekształcone do Δ / Δ/ATR)
    use_residual: bool = False,
    close_idx: Optional[int] = None,
    atr_idx: Optional[int] = None,       # jeśli uczyłeś Δ/ATR
    scoring: str = "mae",
    n_repeats: int = 30,
    random_state: int = 42,
    block_size: Optional[int] = 20,      # sensowny default dla TS
    clip_at_zero: bool = True,
    top_n: Optional[int] = 20,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Wersja „wprost” dla modeli mających .predict(). Przydatna dla DT/RF/XGB/MLP.
    """
    # dopasuj cel (poziom vs residualna przestrzeń)
    if use_residual:
        if close_idx is None:
            raise ValueError("use_residual=True wymaga close_idx.")
        delta = y - X[:, close_idx]
        if atr_idx is not None:
            eps = 1e-8
            y_pi = delta / (X[:, atr_idx] + eps)
        else:
            y_pi = delta
    else:
        y_pi = y

    plot_permutation_importance_generic(
        predict_fn=model.predict,
        X=X, y=y_pi, feature_names=feature_names,
        scoring=scoring, n_repeats=n_repeats, random_state=random_state,
        block_size=block_size, clip_at_zero=clip_at_zero, top_n=top_n,
        title=title or "Permutation importance",
        save_path=save_path,
    )


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
    plot_model_permutation_importance(
        model=model, X=X_test, y=y_test, feature_names=feature_names,
        use_residual=use_residual, close_idx=close_idx, atr_idx=atr_idx,
        scoring="mae", n_repeats=30, block_size=20, clip_at_zero=True, top_n=20,
        title=f"{model_name} – permutation importance (MAE)",
        save_path=os.path.join(
            results_dir, f"{model_name.lower().replace(' ', '_')}_perm_mae.png"
        ),
    )

def plot_lstm_permutation_importance(
    model: Any,
    X_seq: np.ndarray,               # (n_seq, window, n_feat)
    y: np.ndarray,                   # UWAGA: w TEJ SAMEJ skali co predykcja modelu (u Ciebie: skalowane)
    feature_names: Sequence[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    clip_negatives: bool = True,     # mikro-ujemności obcinane do 0
    title: str = "LSTM – permutation importance (MAE)",
    save_path: Optional[str] = None,
) -> None:
    """
    Permutacja po cechach dla LSTM:
    Dla każdej cechy j i każdego kroku czasowego t permutowane są wartości X[:, t, j]
    między sekwencjami (batch), zachowując porządek w czasie w obrębie sekwencji.
    Wartością ważności jest wzrost MAE względem bazowego MAE.

    WAŻNE: y musi być w tej samej skali co wyjście modelu (w Twoim pipeline: to y_te_seq
    przed inverse_transform).
    """
    rng = np.random.default_rng(random_state)

    # baseline na niezmienionych danych
    y_pred_base = model.predict(X_seq, verbose=0).ravel()
    baseline_mae = float(np.mean(np.abs(y - y_pred_base)))

    n_feat = X_seq.shape[2]
    importances = np.zeros(n_feat, dtype=float)

    for j in range(n_feat):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_seq.copy()
            # permutacja kolumny j osobno w każdym kroku czasowym
            for t in range(Xp.shape[1]):
                rng.shuffle(Xp[:, t, j])
            y_pred_perm = model.predict(Xp, verbose=0).ravel()
            mae_perm = float(np.mean(np.abs(y - y_pred_perm)))
            deltas.append(mae_perm - baseline_mae)

        imp = float(np.mean(deltas))
        importances[j] = max(0.0, imp) if clip_negatives else imp

    # rysunek
    order = np.argsort(importances)
    names_sorted = np.array(feature_names)[order]
    vals_sorted  = importances[order]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names_sorted, vals_sorted)
    ax.set_title(title)
    ax.set_xlabel("Wzrost MAE po permutacji")
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def analyze_lstm_permutation_only(
    model: Any,
    X_test_seq: np.ndarray,          # (n_seq, window, n_feat)
    y_test_scaled: np.ndarray,       # UWAGA: skalowane y (ta sama skala co wyjście modelu)
    feature_names: Sequence[str],
    results_dir: str,
    model_name: str = "LSTM",
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> None:
    """
    Wygodny wrapper: jeden call → zapis wykresu permutation importance dla LSTM.
    """
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(
        results_dir, f"{model_name.lower().replace(' ', '_')}_perm_mae.png"
    )
    plot_lstm_permutation_importance(
        model=model,
        X_seq=X_test_seq,
        y=y_test_scaled,             # ta sama skala co wyjście modelu (przed inverse_transform)
        feature_names=feature_names,
        n_repeats=n_repeats,
        random_state=random_state,
        clip_negatives=True,
        title=f"{model_name} – permutation importance (MAE)",
        save_path=save_path,
    )