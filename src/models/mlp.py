from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error


def train_mlp(X_train, y_train):
    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'mlp__activation': ['relu'],
        'mlp__alpha': [0.0001, 0.001],
        'mlp__learning_rate': ['adaptive']
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(max_iter=1000, early_stopping=True, random_state=42))
    ])

    mae = make_scorer(mean_absolute_error)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=mae,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    return {
        'model': 'MLPRegressor (GridSearch, scaled)',
        'MAE mean': grid_search.best_score_,
        'Best params': grid_search.best_params_
    }, grid_search.best_estimator_


def predict_mlp(best_model, X_test):
    return best_model.predict(X_test)