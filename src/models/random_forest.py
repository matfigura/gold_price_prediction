from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

def train_random_forest(X_train, y_train, random_state=42):
    """
    Proste strojenie lasu losowego:
    - TimeSeriesSplit(5),
    - jedna metryka: neg_mean_absolute_error,
    - refit=True (po CV trenowanie na całym train).
    Zwraca (result_dict, best_estimator).
    """
    base = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,        # szybciej na wielu rdzeniach
        bootstrap=True    # standard w RF
    )

    # Zwięzła i łatwa do obrony siatka:
    param_distributions = {
        "n_estimators": [100, 200, 400],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        # losowy podzbiór cech przy podziale (dekorrelacja drzew):
        "max_features": ["sqrt", "log2", None],
        # (opcjonalnie, gdy danych jest dużo – ułamek próbek na drzewo)
        # "max_samples": [None, 0.7, 0.9],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=40,                        # rozsądnie: szybko i skutecznie
        scoring="neg_mean_absolute_error",# 1 metryka → prosto
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    result = {
        "model": "Random Forest (RandomizedSearch, TSCV)",
        "CV MAE": -search.best_score_,           # odwrócenie „neg_” → czytelny MAE
        "Best params": search.best_params_,
    }
    return result, search.best_estimator_


def predict_random_forest(best_model, X_test):
    return best_model.predict(X_test)