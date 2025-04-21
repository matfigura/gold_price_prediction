from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import time


def train_xgboost(X_train, y_train):
    print("[XGBoost] Rozpoczynam strojenie hiperparametrów...")
    start_time = time.time()

    param_grid = {
    'n_estimators': [1000],         
    'learning_rate': [0.001],    
    'max_depth': [3],                     
    'subsample': [0.8],                    
    'colsample_bytree': [0.8, 1.0]              


    }

    base_model = XGBRegressor(random_state=42, verbosity=0)
    mae = make_scorer(mean_absolute_error)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=mae,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    

    grid_search.fit(X_train, y_train)

    end_time = time.time()
    print(f"[XGBoost] Zakończono strojenie w {round(end_time - start_time, 2)} sekundy.")
    print(f"[XGBoost] Najlepsze parametry: {grid_search.best_params_}")
    print(f"[XGBoost] MAE (cross-val): {round(grid_search.best_score_, 3)}")

    return {
        'model': 'XGBoost (GridSearch)',
        'MAE mean': grid_search.best_score_,
        'Best params': grid_search.best_params_
    }, grid_search.best_estimator_

def predict_xgboost(best_model, X_test):
    return best_model.predict(X_test)