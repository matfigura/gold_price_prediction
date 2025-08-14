from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np


def train_decision_tree(
    X_train, y_train,
    cv_splits=5,
    random_state=42,
    n_iter=50,
    refit_metric="neg_mean_absolute_error"
):
    """
    Drzewo regresyjne + RandomizedSearchCV + TimeSeriesSplit.
    Zwraca (result_dict, best_estimator_).
    """
    base_model = DecisionTreeRegressor(random_state=random_state)

    ccp_candidates = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    param_distributions = {
        "max_depth": [3, 5, 7, 10, 20, 50, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha": ccp_candidates,
    }

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # zdefiniuj scoring z pełnymi nazwami + fallback dla RMSE
    rmse_custom = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False
    )
    scoring = {
        "neg_mean_absolute_error": "neg_mean_absolute_error",
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",  # może nie istnieć w starszym sklearn
        "rmse_custom": rmse_custom,  # fallback
        "r2": "r2",
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_metric,   # np. "neg_mean_absolute_error"
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    # pobierz CV RMSE niezależnie od wersji sklearn
    cvres = search.cv_results_
    if "mean_test_neg_root_mean_squared_error" in cvres:
        cv_rmse = -cvres["mean_test_neg_root_mean_squared_error"][search.best_index_]
    else:
        cv_rmse = -cvres["mean_test_rmse_custom"][search.best_index_]

    result = {
        "model": "Decision Tree (RandomizedSearch, TSCV)",
        "CV MAE": -cvres["mean_test_neg_mean_absolute_error"][search.best_index_],
        "CV RMSE": cv_rmse,
        "Best params": search.best_params_,
    }
    return result, search.best_estimator_