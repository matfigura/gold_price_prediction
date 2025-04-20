from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def train_xgboost_custom(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        n_estimators=100000,
        learning_rate=0.001,
        early_stopping_rounds=100,
        eval_metric="mae",
        random_state=42,
        verbosity=1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1000
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    result = {
        'model': 'XGBoost (Early Stopping)',
        'MAE mean': mae,
        'Best params': 'custom (n_estimators=10000, lr=0.001, early_stopping=50)'
    }
    return result, model, y_pred