from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np

def train_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.5],
        'kernel': ['rbf']  # RBF działa dobrze przy nieliniowych zależnościach
    }

    base_model = SVR()
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
        'model': 'SVM (SVR, GridSearch)',
        'MAE mean': grid_search.best_score_,
        'Best params': grid_search.best_params_
    }, grid_search.best_estimator_

def predict_svm(best_model, X_test):
    return best_model.predict(X_test)