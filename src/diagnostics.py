from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from src.models.lstm import build_lstm_model
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore[reportMissingImports]
from sklearn.preprocessing import StandardScaler

# Ścieżka zapisu wykresów
OUTPUT_DIR = "results/curves"

# UTWÓRZ KATALOG, JEŚLI NIE ISTNIEJE
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ————————————————————————————————————————
# 1. Decision Tree (już istniejące)
# ————————————————————————————————————————
def plot_learning_curve_dt(model, X, y, cv=5, scoring='r2', save_path=os.path.join(OUTPUT_DIR, 'learning_curve_dt.png')):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Train', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation', marker='s')
    plt.xlabel('Rozmiar zbioru treningowego')
    plt.ylabel(scoring.upper())
    plt.title('Krzywa uczenia – Decision Tree')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_validation_curve_dt(X, y, param_name, param_range, cv=5, scoring='r2', save_path=os.path.join(OUTPUT_DIR, 'validation_curve_dt.png')):
    model = DecisionTreeRegressor(random_state=42)
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mean, label='Train', marker='o')
    plt.plot(param_range, val_mean, label='Validation', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('R²')
    plt.title(f'Krzywa złożoności – {param_name} (Decision Tree)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ————————————————————————————————————————
# 2. XGBoost (sklearn API)
# ————————————————————————————————————————

def plot_learning_curve_xgb(params, X, y, cv=5, scoring='r2', save_path=os.path.join(OUTPUT_DIR, 'learning_curve_xgb.png')):
    # params: słownik hiperparametrów dla XGBRegressor
    estimator = XGBRegressor(**params)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Train', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation', marker='s')
    plt.xlabel('Rozmiar zbioru treningowego')
    plt.ylabel(scoring.upper())
    plt.title('Krzywa uczenia – XGBoost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_validation_curve_xgb(param_name, param_range, X, y, fixed_params=None, cv=5, scoring='r2', save_path=os.path.join(OUTPUT_DIR, 'validation_curve_xgb.png')):
    # fixed_params: słownik wszystkich parametrów poza tym, którego używamy w param_range
    train_scores = []
    val_scores = []
    for val in param_range:
        params = {} if fixed_params is None else fixed_params.copy()
        params[param_name] = val
        model = XGBRegressor(**params)
        train_scores_fold, val_scores_fold = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=[val],
            cv=cv, scoring=scoring, n_jobs=-1
        )
        train_scores.append(np.mean(train_scores_fold, axis=1)[0])
        val_scores.append(np.mean(val_scores_fold, axis=1)[0])

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores, label='Train', marker='o')
    plt.plot(param_range, val_scores, label='Validation', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('R²')
    plt.title(f'Krzywa złożoności – {param_name} (XGBoost)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curve_xgb_es(
    X_train,
    X_test,
    y_train,
    y_test,
    earlystop_params=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2',
    save_path=os.path.join(OUTPUT_DIR, 'learning_curve_xgb_es.png')
):

    if earlystop_params is None:
        earlystop_params = {
            'n_estimators': 100000,
            'learning_rate': 0.001,
            'early_stopping_rounds': 100,
            'eval_metric': 'mae',
            'random_state': 42,
            'verbosity': 0
        }

    train_scores = []
    val_scores = []
    n_total = X_train.shape[0]

    for frac in train_sizes:
        # Ile próbek bierzemy z X_train w tym kroku?
        n_train = int(n_total * frac)
        X_sub = X_train[:n_train]
        y_sub = y_train[:n_train]

        # Tworzymy model XGBRegressor z przekazanymi parametrami Early Stopping
        model = XGBRegressor(**earlystop_params)

        # Trenujemy na podzbiorze X_sub, a jako walidację podajemy zew. X_test
        model.fit(
            X_sub, y_sub,
            eval_set=[(X_sub, y_sub), (X_test, y_test)],
            verbose=False
        )

        # Predykcja na zbiorze treningowym podzbioru
        y_sub_pred = model.predict(X_sub)
        # Predykcja na pełnym zbiorze walidacyjnym (X_test)
        y_test_pred = model.predict(X_test)

        # Liczymy R² na treningowym i walidacyjnym
        r2_tr = r2_score(y_sub, y_sub_pred)
        r2_val = r2_score(y_test, y_test_pred)

        train_scores.append(r2_tr)
        val_scores.append(r2_val)

    # Średnie (tutaj są pojedyncze wartości, bo każdy punkt to jeden trening)
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)

    # Rysujemy wykres
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, label='Train (R²)', marker='o')
    plt.plot(train_sizes, val_scores, label='Validation (R²)', marker='s')
    plt.xlabel('Frakcja zbioru treningowego')
    plt.ylabel(scoring.upper())
    plt.title('Krzywa uczenia – XGBoost (Early Stopping)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ————————————————————————————————————————
# 3. MLPRegressor
# ————————————————————————————————————————

def plot_learning_curve_mlp(params, X, y, cv=5, scoring='r2', save_path=os.path.join(OUTPUT_DIR, 'learning_curve_mlp.png')):
    estimator = MLPRegressor(**params)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Train', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation', marker='s')
    plt.xlabel('Rozmiar zbioru treningowego')
    plt.ylabel(scoring.upper())
    plt.title('Krzywa uczenia – MLPRegressor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_validation_curve_mlp(param_name, param_range, X, y, fixed_params=None, cv=5, scoring='r2', save_path=os.path.join(OUTPUT_DIR, 'validation_curve_mlp.png')):
    train_scores = []
    val_scores = []
    for val in param_range:
        params = {} if fixed_params is None else fixed_params.copy()
        params[param_name] = val
        model = MLPRegressor(**params)
        train_scores_fold, val_scores_fold = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=[val],
            cv=cv, scoring=scoring, n_jobs=-1
        )
        train_scores.append(np.mean(train_scores_fold, axis=1)[0])
        val_scores.append(np.mean(val_scores_fold, axis=1)[0])

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores, label='Train', marker='o')
    plt.plot(param_range, val_scores, label='Validation', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('R²')
    plt.title(f'Krzywa złożoności – {param_name} (MLPRegressor)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ————————————————————————————————————————
# 4. LSTM – niestandardowa procedura
# ————————————————————————————————————————

def plot_learning_curve_lstm(
    X, y, window_size, build_model_fn,
    epochs=10, batch_size=32,
    train_sizes=np.linspace(0.1, 1.0, 5),
    save_path=os.path.join(OUTPUT_DIR, 'learning_curve_lstm.png')
):
    train_scores = []
    val_scores = []
    n_total = X.shape[0]

    for frac in train_sizes:
        n_train = int(n_total * frac)
        X_sub = X[:n_train]
        y_sub = y[:n_train]
        split = int(n_train * 0.8)
        X_tr, X_val = X_sub[:split], X_sub[split:]
        y_tr, y_val = y_sub[:split], y_sub[split:]

        model = build_model_fn((window_size, X.shape[2]))
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)
        r2_tr = r2_score(y_tr, y_tr_pred)
        r2_val = r2_score(y_val, y_val_pred)
        train_scores.append(r2_tr)
        val_scores.append(r2_val)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, label='Train', marker='o')
    plt.plot(train_sizes, val_scores, label='Validation', marker='s')
    plt.xlabel('Frakcja zbioru treningowego')
    plt.ylabel('R²')
    plt.title('Krzywa uczenia – LSTM')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_validation_curve_lstm(X, y, window_size, build_model_fn, param_name, param_range,
                               epochs=10, batch_size=32, save_path=os.path.join(OUTPUT_DIR, 'validation_curve_lstm.png')):
    train_scores = []
    val_scores = []
    for val in param_range:
        # Załóżmy, że build_model_fn przyjmuje parametr val jako wartość hipermetra
        model = build_model_fn((window_size, X.shape[2]), **{param_name: val})
        split = int(X.shape[0] * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)
        r2_tr = r2_score(y_tr, y_tr_pred)
        r2_val = r2_score(y_val, y_val_pred)
        train_scores.append(r2_tr)
        val_scores.append(r2_val)

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores, label='Train', marker='o')
    plt.plot(param_range, val_scores, label='Validation', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('R²')
    plt.title(f'Krzywa złożoności – {param_name} (LSTM)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
