from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    base_model = RandomForestRegressor(random_state=42)
    mae = make_scorer(mean_absolute_error)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=mae,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return {
        'model': 'Random Forest (GridSearch)',
        'MAE mean': grid_search.best_score_,
        'Best params': grid_search.best_params_
    }, grid_search.best_estimator_

def predict_random_forest(best_model, X_test):
    return best_model.predict(X_test)
