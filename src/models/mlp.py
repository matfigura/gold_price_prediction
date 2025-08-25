from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

def train_mlp(
    X_train, y_train,
    random_state=42,
    cv_splits=5,
    n_iter=30
):
    """
    MLPRegressor:
    - Pipeline: StandardScaler -> MLP
    - TSCV (TimeSeriesSplit) + RandomizedSearchCV
    - scoring: MAE (neg_mean_absolute_error)
    Zwraca (result_dict, best_estimator_).
    """
    print("[MLP] ▶ Start strojenia (RandomizedSearchCV + TimeSeriesSplit)")
    print(f"[MLP]   Rozmiar TRAIN: {X_train.shape[0]} próbek, {X_train.shape[1]} cech")
    t0 = time.time()

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
            max_iter=2000,          # duży zapas
            tol=1e-4,               # kryterium zbieżności
            shuffle=False,          # ważne przy szeregach!
            early_stopping=False,   # bez wewnętrznego walidowania (chronologia)
            random_state=random_state,
            verbose=False
        ))
    ])

    # zwięzła, praktyczna przestrzeń:
    param_distributions = {
        'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
        'mlp__learning_rate_init': [0.001, 0.003, 0.01],
        'mlp__batch_size': ['auto', 32, 64, 128],
    }

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(X_train, y_train)
    t1 = time.time()

    result = {
        'model': 'MLPRegressor (RandomizedSearch, scaled, TSCV)',
        'CV MAE': -search.best_score_,
        'Best params': search.best_params_
    }
    print(f"[MLP] ✅ Zakończono strojenie w {t1 - t0:.2f}s")
    print(f"[MLP]    Najlepsze parametry: {result['Best params']}")
    print(f"[MLP]    CV MAE (↓): {result['CV MAE']:.6f}")
    return result, search.best_estimator_


def predict_mlp(best_model, X_test):
    return best_model.predict(X_test)