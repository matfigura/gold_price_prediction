from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np

def train_random_forest_randomized(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    base_model = RandomForestRegressor(random_state=42)
    mae = make_scorer(mean_absolute_error)

    randomized_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=100,  # liczba losowych prób
        scoring=mae,
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    randomized_search.fit(X_train, y_train)

    return {
        'model': 'Random Forest (RandomizedSearch)',
        'MAE mean': randomized_search.best_score_,
        'Best params': randomized_search.best_params_
    }, randomized_search.best_estimator_

def predict_rf_randomized(best_model, X_test):
    return best_model.predict(X_test)