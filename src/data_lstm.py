import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import add_core_indicators, add_basic_lags

def _build_features(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Zwraca X_df zgodnie z presetem jak w prepare_data()."""
    tmp = df.copy()

    if dataset_type == "ohlc":
        X_df = tmp[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    elif dataset_type == "ta_lags":
        tmp = add_core_indicators(tmp)
        tmp = add_basic_lags(tmp, lag_list=(1, 5, 10), add_returns=True)
        X_df = tmp[['Close', 'Volume',
                    'macd_diff', 'rsi_14', 'bb_percent', 'bb_width', 'atr_14',
                    'sma_20', 'ema_12', 'cci_20',
                    'close_lag_1', 'close_lag_5', 'close_lag_10',
                    'ret_1d', 'ret_5d']].copy()

    elif dataset_type == "mixed":
        tmp = add_core_indicators(tmp)
        tmp = add_basic_lags(tmp, lag_list=(1, 5, 10), add_returns=True)
        X_df = tmp[['Open', 'High', 'Low', 'Close', 'Volume',
                    'macd_diff', 'rsi_14', 'bb_percent', 'bb_width', 'atr_14',
                    'sma_20', 'ema_12', 'cci_20',
                    'close_lag_1', 'close_lag_5', 'close_lag_10',
                    'ret_1d', 'ret_5d']].copy()
    else:
        raise ValueError("dataset_type ∈ {'ohlc','ta_lags','mixed'}")

    return X_df


def create_lstm_data(
    file_path: str = 'data/gold_data.csv',
    dataset_type: str = 'ta_lags',     # spójnie z prepare_data()
    target_mode: str = 'close_next',   # 'close_next' albo 'return_next'
    window_size: int = 10,
    test_size: float = 0.2,
    scale_X: bool | None = None        # None => auto (dla LSTM domyślnie True)
):
    """
    Buduje sekwencje dla LSTM na tych samych presetach co prepare_data().
    - Chronologiczny split (80/20 domyślnie),
    - Skalowanie TYLKO na train (X oraz y),
    - Sekwencje o długości `window_size`, bez tasowania.

    Zwraca:
      X_train: (n_train_seq, window_size, n_feat)
      X_test : (n_test_seq , window_size, n_feat)
      y_train: (n_train_seq,)
      y_test : (n_test_seq,)
      dates_test: daty odpowiadające elementom y_test
      scaler_y: StandardScaler do odwracania skali y (gdy target_mode='close_next')
      feature_names: lista nazw cech
    """
    # 1) Wczytanie i porządkowanie
    df = pd.read_csv(file_path, sep=';')
    for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"Brakuje wymaganej kolumny: {col}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # 2) Target (jak w prepare_data)
    if target_mode == 'close_next':
        df['target'] = df['Close'].shift(-1)
    elif target_mode == 'return_next':
        df['ret_1d_now'] = df['Close'].pct_change(1)
        df['target'] = df['ret_1d_now'].shift(-1)
    else:
        raise ValueError("target_mode ∈ {'close_next','return_next'}")

    # 3) Cechy wg presetu (jak w prepare_data)
    X_df = _build_features(df, dataset_type)

    # 4) Złożenie ramki i drop NaN (po wskaźnikach/lagach i shifcie targetu)
    work = pd.concat([df[['Date', 'target']], X_df], axis=1).dropna().reset_index(drop=True)
    dates = work['Date'].values
    y = work['target'].values
    X = work.drop(columns=['Date', 'target'])
    feature_names = list(X.columns)

    # 5) Chronologiczny split (80/20 domyślnie)
    n = len(work)
    split_idx = int(n * (1 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Niepoprawny split — sprawdź liczność danych.")

    # 6) Skalowanie (LSTM zwykle wymaga; auto=True gdy None)
    if scale_X is None:
        scale_X = True

    if scale_X:
        scaler_x = StandardScaler().fit(X.iloc[:split_idx])             # fit tylko na train
        X_scaled = scaler_x.transform(X)
    else:
        scaler_x = None
        X_scaled = X.values

    scaler_y = StandardScaler().fit(y[:split_idx].reshape(-1, 1))       # fit tylko na train (y)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    # 7) Generowanie sekwencji po CAŁOŚCI (granica wyznaczy split)
    X_seq, y_seq, dates_seq = [], [], []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i + window_size])
        y_seq.append(y_scaled[i + window_size])
        dates_seq.append(dates[i + window_size])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    dates_seq = np.array(dates_seq)

    # 8) Granica sekwencji: pierwszy indeks testu = split_idx - window_size
    boundary = split_idx - window_size
    if boundary <= 0:
        raise ValueError(
            f"Okno ({window_size}) zbyt duże względem części treningowej ({split_idx}). "
            f"Zmień window_size lub test_size."
        )

    X_train, y_train = X_seq[:boundary], y_seq[:boundary]
    X_test,  y_test  = X_seq[boundary:], y_seq[boundary:]
    dates_test = dates_seq[boundary:]

    return X_train, X_test, y_train, y_test, dates_test, scaler_y, feature_names