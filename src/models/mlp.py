from typing import Dict, Any, Tuple, Optional   
import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_mlp(
    X_train, y_train,
    random_state: int = 42,
    cv_splits: int = 5,
    n_iter: int = 30,
    fast: bool = False,            
    cv_gap: int = 0,               
    tag: str = "",                 
    fix_hidden_size: Optional[tuple] = None,  
):


    def _make_pipe(max_iter: int, tol: float) -> Pipeline:
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                batch_size=64,
                alpha=1e-4,
                max_iter=max_iter,
                tol=tol,
                shuffle=False,                 
                early_stopping=False,          
                validation_fraction=0.10,      
                n_iter_no_change=20,           
                random_state=random_state,
                verbose=False
            ))
        ])

    def _tscv(n_splits: int, gap: int) -> TimeSeriesSplit:
        
        try:
            return TimeSeriesSplit(n_splits=n_splits, gap=gap)
        except TypeError:
            return TimeSeriesSplit(n_splits=n_splits)


    scan_cv = 3                                    
    scan_iter = min(n_iter, 12)                    
    scan_max_iter = 400                            
    scan_tol = 1e-4                                
    scan_params = {                                
        'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32)],
        'mlp__alpha': [1e-6, 1e-5, 1e-4],
        'mlp__learning_rate_init': [5e-4, 1e-3, 3e-3],
        'mlp__batch_size': [32, 64],
        'mlp__activation': ['relu'],
    }


    fin_max_iter = 800                             
    fin_tol = 1e-4                                 

    # Log startu
    print("[MLP] ▶ Start strojenia (RandomizedSearchCV + TimeSeriesSplit)")
    print(f"[MLP]   TRAIN: {X_train.shape[0]} próbek, {X_train.shape[1]} cech | fast={fast}"
          f"{' | ' + tag if tag else ''}")
    t_global0 = time.time()



    if fast:
        t0 = time.time()
        pipe_scan = _make_pipe(scan_max_iter, scan_tol)
        search_scan = RandomizedSearchCV(
            estimator=pipe_scan,
            param_distributions=scan_params,
            n_iter=scan_iter,
            scoring='neg_mean_absolute_error',
            cv=_tscv(scan_cv, cv_gap),
            random_state=random_state,
            n_jobs=-1,
            refit=True,
            verbose=1,
            return_train_score=False
        )
        print("[MLP] ▶ Etap 1/1 (fast) — skan rozmiarów")
        search_scan.fit(X_train, y_train)
        t1 = time.time()

        result = {
            'model': f"MLPRegressor (RandomizedSearch, scaled, TSCV fast)",
            'CV MAE': -search_scan.best_score_,
            'Best params': search_scan.best_params_
        }
        if tag:
            result['Tag'] = tag
        print(f"[MLP] ✅ Zakończono (fast) w {t1 - t0:.2f}s | CV MAE={result['CV MAE']:.6f}"
              f"{' | ' + tag if tag else ''}")
        print(f"[MLP]    Najlepsze parametry: {result['Best params']}")
        return result, search_scan.best_estimator_

  
    if fix_hidden_size is not None:
        t0 = time.time()
        pipe_fin = _make_pipe(fin_max_iter, fin_tol)
        fin_params = {
            'mlp__hidden_layer_sizes': [fix_hidden_size],   
            'mlp__alpha': [1e-4, 1e-3],
            'mlp__learning_rate_init': [5e-4, 1e-3, 3e-3],
            'mlp__batch_size': [32],
            'mlp__activation': ['relu'],
        }
        search_fin = RandomizedSearchCV(
            estimator=pipe_fin,
            param_distributions=fin_params,
            n_iter=n_iter,
            scoring='neg_mean_absolute_error',
            cv=_tscv(cv_splits, cv_gap),
            random_state=random_state,
            n_jobs=-1,
            refit=True,
            verbose=1,
            return_train_score=False
        )
        print(f"[MLP] ▶ Etap 2 (fixed HLS={fix_hidden_size}) — dostrajanie")
        search_fin.fit(X_train, y_train)
        t1 = time.time()

        result = {
            'model': f"MLPRegressor (RandomizedSearch, scaled, TSCV)",
            'CV MAE': -search_fin.best_score_,
            'Best params': search_fin.best_params_
        }
        if tag:
            result['Tag'] = tag + f" | fix_hls={fix_hidden_size}"
        print(f"[MLP] ✅ Zakończono (etap 2) w {t1 - t0:.2f}s | CV MAE={result['CV MAE']:.6f}"
              f"{' | ' + result.get('Tag','') if tag else ''}")
        print(f"[MLP]    Najlepsze parametry: {result['Best params']}")
        return result, search_fin.best_estimator_


    t1_0 = time.time()
    pipe_scan = _make_pipe(scan_max_iter, scan_tol)
    search_scan = RandomizedSearchCV(
        estimator=pipe_scan,
        param_distributions=scan_params,
        n_iter=scan_iter,
        scoring='neg_mean_absolute_error',
        cv=_tscv(scan_cv, cv_gap),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=False
    )
    print("[MLP] ▶ Etap 1/2 — skan rozmiarów")
    search_scan.fit(X_train, y_train)
    t1_1 = time.time()
    best_hls = search_scan.best_params_['mlp__hidden_layer_sizes']  
    print(f"[MLP] ✅ Etap 1/2 done w {t1_1 - t1_0:.2f}s | best_hls={best_hls} | CV MAE={-search_scan.best_score_:.6f}"
          f"{' | ' + tag if tag else ''}")

 
    t2_0 = time.time()
    pipe_fin = _make_pipe(fin_max_iter, fin_tol)
    fin_params = {
        'mlp__hidden_layer_sizes': [best_hls],         
        'mlp__alpha': [1e-4, 1e-3],
        'mlp__learning_rate_init': [5e-4, 1e-3, 3e-3],
        'mlp__batch_size': [32],
        'mlp__activation': ['relu'],
    }
    search_fin = RandomizedSearchCV(
        estimator=pipe_fin,
        param_distributions=fin_params,
        n_iter=n_iter,                                
        scoring='neg_mean_absolute_error',
        cv=_tscv(cv_splits, cv_gap),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=False
    )
    print("[MLP] ▶ Etap 2/2 — dostrajanie wokół best_hls")
    search_fin.fit(X_train, y_train)
    t2_1 = time.time()

  
    result = {
        'model': f"MLPRegressor (RandomizedSearch, scaled, TSCV)",
        'CV MAE': -search_fin.best_score_,
        'Best params': search_fin.best_params_
    }
    if tag:
        result['Tag'] = tag + f" | stage1_hls={best_hls}"
    print(f"[MLP] ✅ Zakończono strojenie w {time.time() - t_global0:.2f}s | CV MAE={result['CV MAE']:.6f}"
          f"{' | ' + result.get('Tag','') if tag else ''}")
    print(f"[MLP]    Najlepsze parametry: {result['Best params']}")
    return result, search_fin.best_estimator_


def predict_mlp(best_model, X_test):
    
    return best_model.predict(X_test)