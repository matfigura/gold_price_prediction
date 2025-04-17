from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np

def train_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    base_model = DecisionTreeRegressor(random_state=42)
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
        'model': 'Decision Tree (GridSearch)',
        'MAE mean': grid_search.best_score_,
        'Best params': grid_search.best_params_
    }, grid_search.best_estimator_

def predict_decision_tree(best_model, X_test):
    return best_model.predict(X_test)




