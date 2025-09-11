import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, SMAIndicator, CCIIndicator

PRACA_CFG = dict(
    sma_n=10,
    wma_n=14,
    mom_n=10,
    stoch_n=14,
    stoch_smooth=3,
    rsi_n=14,
    wr_n=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    cci_n=20
)

def weighted_moving_average(s: pd.Series, window: int) -> pd.Series:
    w = np.arange(1, window + 1)
    return s.rolling(window).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def add_indicators_praca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma_10'] = SMAIndicator(close=df['Close'], window=PRACA_CFG['sma_n']).sma_indicator()
    df['wma_14'] = weighted_moving_average(df['Close'], PRACA_CFG['wma_n'])
    n = PRACA_CFG['mom_n']
    df['mom_10'] = df['Close'] - df['Close'].shift(n - 1)
    stoch = StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=PRACA_CFG['stoch_n'], smooth_window=PRACA_CFG['stoch_smooth']
    )
    df['stoch_k_14'] = stoch.stoch()
    df['stoch_d_14'] = stoch.stoch_signal()
    df['rsi_14'] = RSIIndicator(close=df['Close'], window=PRACA_CFG['rsi_n']).rsi()
    df['wr_14'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=PRACA_CFG['wr_n']).williams_r()
    macd = MACD(close=df['Close'], window_fast=PRACA_CFG['macd_fast'], window_slow=PRACA_CFG['macd_slow'], window_sign=PRACA_CFG['macd_signal'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['cci_20'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=PRACA_CFG['cci_n'], constant=0.015).cci()
    return df

def prepare_data(
    file_path: str = 'data/gold_data.csv',
    dataset_type: str = "ta_research_set",
    test_size: float = 0.2,
    scale_X: bool | None = None,
    target_mode: str = "close_next"
):
    df = pd.read_csv(file_path, sep=';')
    if 'Date' not in df.columns:
        raise ValueError("Brak kolumny 'Date'.")
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"Brak wymaganej kolumny: {col}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    if target_mode == "close_next":
        df['target'] = df['Close'].shift(-1)
    elif target_mode == "return_next":
        df['ret_1d_now'] = df['Close'].pct_change(1)
        df['target'] = df['ret_1d_now'].shift(-1)
    else:
        raise ValueError("target_mode ∈ {'close_next','return_next'}")

    if dataset_type == "ohlc":
        X_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    elif dataset_type == "mixed":
        tmp = add_indicators_praca(df).copy()
        tmp[['Open', 'High', 'Low']] = df[['Open', 'High', 'Low']]
        X_df = tmp[['Open','High','Low','Close','Volume','sma_10','wma_14','mom_10','stoch_k_14','stoch_d_14','rsi_14','wr_14','macd','macd_signal','cci_20']].copy()
    elif dataset_type == "ta_research_set":
        tmp = add_indicators_praca(df)
        X_df = tmp[['Close','Volume','sma_10','wma_14','mom_10','stoch_k_14','stoch_d_14','rsi_14','wr_14','macd','macd_signal','cci_20']].copy()
    elif dataset_type in {"ta_research_set_RF", "ta_research_set_DT", "ta_research_set_XGB"}:
        tmp = add_indicators_praca(df)
        X_df = tmp[['Close']].copy()
    elif dataset_type == "ohlc_MLP":
        tmp = add_indicators_praca(df)
        X_df = tmp[['Open','High','Low','Close']].copy()
    else:
        raise ValueError("Niewspierany dataset_type.")

    df_full = pd.concat([df[['Date','target']], X_df], axis=1).dropna().reset_index(drop=True)
    dates = df_full['Date'].values
    y = df_full['target'].values
    X = df_full.drop(columns=['Date','target'])

    n = len(df_full)
    split_idx = int(n * (1 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Niepoprawny split – za mało danych.")
    X_train_df, X_test_df = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    if scale_X is None:
        scale_X = dataset_type not in {"ohlc", "ohlc_MLP"}

    scaler = None
    if scale_X:
        scaler = StandardScaler().fit(X_train_df)
        X_train = scaler.transform(X_train_df)
        X_test = scaler.transform(X_test_df)
    else:
        X_train = X_train_df.values
        X_test = X_test_df.values

    feature_names = list(X.columns)
    return X_train, X_test, y_train, y_test, dates_test, feature_names, scaler
