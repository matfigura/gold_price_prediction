from typing import Dict, Any, Tuple
import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score


def train_mlp(
    X_train, y_train,
    random_state: int = 42,
    cv_splits: int = 5,
    n_iter: int = 30,
    fast: bool = False,         # [DODANE] tryb szybki jak w DT/RF/XGB
    cv_gap: int = 0,            # [DODANE] przerwa między foldami (chroni przed micro-leakage)
):
    """
    MLPRegressor:
    - Pipeline: StandardScaler -> MLP
    - TimeSeriesSplit (+ opcjonalny gap)
    - RandomizedSearchCV ze scoringiem: MAE (neg_mean_absolute_error)
    Zwraca (result_dict, best_estimator_).

    Tryb `fast=True`:
      • mniejsza liczba foldów (3),
      • mniej kandydatów (min(n_iter, 12)),
      • ciaśniejsza siatka wokół Twoich najlepszych hiperparametrów:
          hidden_layer_sizes=(64,), activation='relu', alpha≈1e-5, lr_init≈0.001, batch_size=64
      • mniejsze max_iter i luźniejsze tol.
    """

    print("[MLP] ▶ Start strojenia (RandomizedSearchCV + TimeSeriesSplit)")
    print(f"[MLP]   TRAIN: {X_train.shape[0]} próbek, {X_train.shape[1]} cech | fast={fast}")
    t0 = time.time()

    # [FAST] korekty budżetu obliczeń
    if fast:
        cv_splits = max(3, min(cv_splits, 3))
        n_iter = min(n_iter, 12)
        max_iter = 800
        tol = 1e-3
    else:
        max_iter = 2000
        tol = 1e-4

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            learning_rate='constant',
            learning_rate_init=0.003,
            batch_size='auto',
            alpha=1e-4,             # L2
            max_iter=max_iter,      # [FAST] niższe w fast
            tol=tol,                # [FAST] luźniejsze w fast
            shuffle=False,          # ważne przy szeregach!
            early_stopping=False,   # bez losowej walidacji wewnętrznej
            random_state=random_state,
            verbose=False
        ))
    ])

    # Siatka strojenia
    if fast:
        # [FAST] wąsko wokół Twoich best params:
        # {'mlp__learning_rate_init': 0.001, 'mlp__hidden_layer_sizes': (64,),
        #  'mlp__batch_size': 64, 'mlp__alpha': 1e-05, 'mlp__activation': 'relu'}
        param_distributions = {
            'mlp__hidden_layer_sizes': [(64,), (64, 32)],     # małe, szybkie
            'mlp__activation': ['relu'],                      # wokół najlepszego
            'mlp__alpha': [1e-6, 1e-5, 1e-4],
            'mlp__learning_rate_init': [0.0005, 0.001, 0.003],
            'mlp__batch_size': [64, 128],                     # wąsko wokół 64
        }
    else:
        # pełniejsza przestrzeń
        param_distributions = {
            'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            'mlp__learning_rate_init': [0.0005, 0.001, 0.003, 0.01],
            'mlp__batch_size': ['auto', 32, 64, 128],
        }

    # TimeSeriesSplit z gap (jeśli wersja sklearn wspiera)
    try:
        tscv = TimeSeriesSplit(n_splits=cv_splits, gap=cv_gap)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',   # MAE
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=False
    )

    search.fit(X_train, y_train)
    t1 = time.time()

    result = {
        'model': f"MLPRegressor (RandomizedSearch, scaled, TSCV{' fast' if fast else ''})",
        'CV MAE': -search.best_score_,                 # MAE ↓
        'Best params': search.best_params_
    }
    print(f"[MLP] ✅ Zakończono strojenie w {t1 - t0:.2f}s | CV MAE={result['CV MAE']:.6f}")
    print(f"[MLP]    Najlepsze parametry: {result['Best params']}")
    return result, search.best_estimator_


def predict_mlp(best_model, X_test):
    return best_model.predict(X_test)