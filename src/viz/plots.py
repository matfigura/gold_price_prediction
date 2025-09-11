from __future__ import annotations

import os
import re
import glob
import math
from typing import Optional, Sequence, Any, Callable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from PIL import Image, ImageDraw, ImageFont


PLOTS_DIR = os.path.join("results", "plots")


try:
    from src.models.xgboost_model import predict_xgboost as _xgb_predict_level
except Exception:
    _xgb_predict_level = None



def _align_feature_names(feature_names: Optional[Sequence[str]], n_feat: int) -> np.ndarray:
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



class XGBLevelWrapper:
    def __init__(self, model: Any):
        self.model = model
    def predict(self, X: np.ndarray) -> np.ndarray:
        if _xgb_predict_level is None:
            return self.model.predict(X)
        return _xgb_predict_level(self.model, X)



def plot_predictions(y_true, y_pred, dates, model_name: str, save_path: str):
    plt.figure(figsize=(10, 5))

    sorted_data = sorted(zip(dates, y_true, y_pred), key=lambda x: pd.to_datetime(x[0]))
    dates_sorted, y_true_sorted, y_pred_sorted = zip(*sorted_data)

    plt.plot(dates_sorted, y_true_sorted, label="Rzeczywiste", linewidth=2)
    plt.plot(dates_sorted, y_pred_sorted, label="Predykcja", linestyle='--', linewidth=2)

    plt.xlabel("Data")
    plt.ylabel("Cena zamknięcia")
    plt.title(f"{model_name} – rzeczywiste vs predykcja")
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()



def build_collage(folder: str = PLOTS_DIR, pattern: str = "*_plot_*.png", cols: int = 3, out_name: str = "_collage.png"):
    os.makedirs(folder, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        print(f"[build_collage] Brak plików dopasowanych do wzorca: {os.path.join(folder, pattern)}")
        return

    images, labels = [], []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")

            img = img.resize((640, 360))
            images.append(img)
            labels.append(os.path.basename(p).replace("_plot_", " | ").replace(".png", ""))
        except Exception as e:
            print(f"[build_collage] Pominięto {p}: {e}")

    if not images:
        print("[build_collage] Brak poprawnych obrazów do kolażu.")
        return

    rows = math.ceil(len(images) / cols)
    thumb_w, thumb_h = images[0].size
    labeled_h = thumb_h + 30
    collage = Image.new("RGB", (cols * thumb_w, rows * labeled_h), (255, 255, 255))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, img in enumerate(images):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * labeled_h
        collage.paste(img, (x, y))

        draw = ImageDraw.Draw(collage)
        label = labels[idx]
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = len(label) * 6, 10  
        draw.text((x + (thumb_w - tw)//2, y + thumb_h + 5), label, fill=(0, 0, 0), font=font)

    out_path = os.path.join(folder, out_name)
    collage.save(out_path)
    print(f"[build_collage] Zapisano kolaż: {out_path}")



def permutation_importance_generic(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    scoring: str = "mae",
    n_repeats: int = 20,
    random_state: int = 42,
    block_size: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    def _score(y_true, y_pred):
        if scoring == "rmse":
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return float(np.mean(np.abs(y_true - y_pred)))

    X = X.values if hasattr(X, "values") else np.asarray(X)
    y = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()

    base_pred = predict_fn(X)
    base_err  = _score(y, base_pred)

    if X.ndim == 2:
        n, d = X.shape
    elif X.ndim == 3:
        n, _, d = X.shape
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
    clip_at_zero: bool = False,
    top_n: Optional[int] = None,
    title: str = "Permutation importance",
    save_path: Optional[str] = None,
):
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
    eval_space: Optional[str] = None,
    pred_kind: str = "auto",
) -> pd.DataFrame:
    X = X.values if hasattr(X, "values") else np.asarray(X)
    y = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()
    _, p = X.shape

    feature_names = _align_feature_names(feature_names, p)
    rng = np.random.default_rng(random_state)

    if close_idx is None:
        close_idx = getattr(model, "_close_idx", None)
    scaler = getattr(model, "_scaler", None)

    def _get_close(vecX: np.ndarray) -> np.ndarray:
        if close_idx is None:
            raise ValueError("Brak close_idx (ani model._close_idx). Nie mogę rekonstruować poziomu z return.")
        c = vecX[:, close_idx]
        if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            c = c * scaler.scale_[close_idx] + scaler.mean_[close_idx]
        return c

    yhat_raw = model.predict(X)
    pred_kind_auto = "level"
    if pred_kind == "auto":
        try:
            c = _get_close(X)
            return_ref = (y - c) / (c + 1e-8)
            err_to_level  = float(np.mean(np.abs(y - yhat_raw)))
            err_to_return = float(np.mean(np.abs(return_ref - yhat_raw)))
            pred_kind_auto = "return" if err_to_return < err_to_level else "level"
        except Exception:
            pred_kind_auto = "level"
    elif pred_kind in ("level", "return"):
        pred_kind_auto = pred_kind
    else:
        raise ValueError("pred_kind must be 'auto' | 'level' | 'return'.")

    def predict_level(X_like: np.ndarray) -> np.ndarray:
        yp = model.predict(X_like)
        if pred_kind_auto == "level":
            return yp
        c = _get_close(X_like)
        return c * (1.0 + yp)

    if scoring == "rmse":
        def _score(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    else:
        def _score(a, b): return float(np.mean(np.abs(a - b)))

    y_pred0 = predict_level(X)
    base_err = _score(y, y_pred0)

    def permute_vec(v: np.ndarray) -> np.ndarray:
        v = v.copy()
        if block_size is None or block_size <= 1:
            rng.shuffle(v)
            return v
        idx = np.arange(len(v))
        blocks = [idx[i:i+block_size] for i in range(0, len(v), block_size)]
        order = rng.permutation(len(blocks))
        return v[np.concatenate([blocks[k] for k in order])]

    inc = np.zeros(p, dtype=float)
    for j in range(p):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = permute_vec(Xp[:, j])
            y_hat_p = predict_level(Xp)
            deltas.append(_score(y, y_hat_p) - base_err)
        inc[j] = float(np.mean(deltas))

    if clip_at_zero:
        inc = np.maximum(inc, 0.0)

    order = np.argsort(inc)[::-1]
    feat_sorted = feature_names[order]
    imp_sorted  = inc[order]
    if top_n is not None:
        feat_sorted = feat_sorted[:top_n]
        imp_sorted  = imp_sorted[:top_n]

    plt.figure(figsize=(12, 6))
    plt.barh(feat_sorted[::-1], imp_sorted[::-1])
    plt.xlabel(f"Wzrost błędu po permutacji ({scoring.upper()}) – skala POZIOMU")
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return pd.DataFrame({"feature": feat_sorted, "importance": imp_sorted})



def plot_lstm_permutation_importance(
    model: Any,
    X_seq: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    clip_negatives: bool = True,
    title: str = "LSTM – permutation importance (MAE)",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    X_seq = np.asarray(X_seq)
    y = np.asarray(y).ravel()
    n_feat = X_seq.shape[2]
    feature_names = _align_feature_names(feature_names, n_feat)

    y_pred_base = model.predict(X_seq, verbose=0).ravel()
    baseline_mae = float(np.mean(np.abs(y - y_pred_base)))

    importances = np.zeros(n_feat, dtype=float)

    for j in range(n_feat):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_seq.copy()
            for t in range(Xp.shape[1]):
                rng.shuffle(Xp[:, t, j])
            y_pred_perm = model.predict(Xp, verbose=0).ravel()
            mae_perm = float(np.mean(np.abs(y - y_pred_perm)))
            deltas.append(mae_perm - baseline_mae)
        imp = float(np.mean(deltas))
        importances[j] = max(0.0, imp) if clip_negatives else imp

    order = np.argsort(importances)[::-1]
    names_sorted = feature_names[order]
    vals_sorted  = importances[order]

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
        use_residual=False,
        close_idx=close_idx, atr_idx=None,
        scoring="mae", n_repeats=30, block_size=20, clip_at_zero=True, top_n=20,
        title=f"{model_name} – permutation importance (MAE, poziom)",
        save_path=os.path.join(results_dir, f"{slug}_perm_mae.png"),
        eval_space=None, pred_kind="auto",
    )


def analyze_lstm_permutation_only(
    model: Any,
    X_test_seq: np.ndarray,
    y_test_scaled: np.ndarray,
    feature_names: Sequence[str],
    results_dir: str,
    model_name: str = "LSTM",
    *,
    n_repeats: int = 10,
    random_state: int = 42,
):
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



__all__ = [
    "PLOTS_DIR",
    "plot_predictions",
    "build_collage",
    "permutation_importance_generic",
    "plot_permutation_importance_generic",
    "plot_model_permutation_importance",
    "plot_lstm_permutation_importance",
    "analyze_permutation_only",
    "analyze_lstm_permutation_only",
    "XGBLevelWrapper",
]
