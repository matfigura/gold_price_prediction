import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, CCIIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# ───────────────────────────────────────────────────────────────
# WMA: ważona średnia krocząca (rosnące wagi 1..n)
# ───────────────────────────────────────────────────────────────
def weighted_moving_average(s: pd.Series, window: int) -> pd.Series:
    w = np.arange(1, window + 1)
    return s.rolling(window).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

# ───────────────────────────────────────────────────────────────
# Zestaw 'praca' – wskaźniki z artykułu (okna wg opisu)
# ───────────────────────────────────────────────────────────────
PRACA_CFG = dict(
    sma_n=10,           # Simple 10-day MA
    wma_n=14,           # Weighted 14-day MA
    mom_n=10,           # Momentum: Ct - C_{t-n+1} (tu n=10)
    stoch_n=14,         # %K,%D (okno 14), smoothing 3
    stoch_smooth=3,
    rsi_n=14,           # RSI(14)
    wr_n=14,            # Williams %R(14)
    macd_fast=12,       # MACD(12,26,9) — współczynniki 0.15/0.075 ↔ okna 12/26
    macd_slow=26,
    macd_signal=9,
    cci_n=20            # CCI(20), constant=0.015
)

def add_indicators_praca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) SMA(10)
    df['sma_10'] = SMAIndicator(close=df['Close'], window=PRACA_CFG['sma_n']).sma_indicator()

    # 2) WMA(14)
    df['wma_14'] = weighted_moving_average(df['Close'], PRACA_CFG['wma_n'])

    # 3) Momentum: Ct - C_{t-n+1}
    n = PRACA_CFG['mom_n']
    df['mom_10'] = df['Close'] - df['Close'].shift(n - 1)

    # 4-5) Stochastic %K/%D (14; smoothing=3)
    stoch = StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=PRACA_CFG['stoch_n'], smooth_window=PRACA_CFG['stoch_smooth']
    )
    df['stoch_k_14'] = stoch.stoch()
    df['stoch_d_14'] = stoch.stoch_signal()

    # 6) RSI(14)
    df['rsi_14'] = RSIIndicator(close=df['Close'], window=PRACA_CFG['rsi_n']).rsi()

    # 7) Williams %R(14)
    df['wr_14'] = WilliamsRIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], lbp=PRACA_CFG['wr_n']
    ).williams_r()

    # 8) MACD(12,26,9)
    macd = MACD(
        close=df['Close'],
        window_fast=PRACA_CFG['macd_fast'],
        window_slow=PRACA_CFG['macd_slow'],
        window_sign=PRACA_CFG['macd_signal']
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # 9) CCI(20) z constant=0.015
    df['cci_20'] = CCIIndicator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=PRACA_CFG['cci_n'], constant=0.015
    ).cci()

    return df


# ───────────────────────────────────────────────────────────────
# 1) LAGI i ZWROTY
# ───────────────────────────────────────────────────────────────
def add_basic_lags(df: pd.DataFrame,
                   lag_list=(1, 2, 3, 5, 10, 20),
                   add_returns=True) -> pd.DataFrame:
    df = df.copy()
    for L in lag_list:
        df[f'close_lag_{L}'] = df['Close'].shift(L)
    if add_returns:
        df['ret_1d']  = df['Close'].pct_change(1)
        df['ret_5d']  = df['Close'].pct_change(5)
        df['ret_10d'] = df['Close'].pct_change(10)
    return df


# ───────────────────────────────────────────────────────────────
# 2) RDZENIOWE WSKAŹNIKI TECHNICZNE (bez look-ahead)
# ───────────────────────────────────────────────────────────────
def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MACD i różnica (trend+momentum)
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd']        = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff']   = df['macd'] - df['macd_signal']

    # RSI(14)
    df['rsi_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # Bollinger %B i szerokość pasma
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    lower = bb.bollinger_lband()
    upper = bb.bollinger_hband()
    df['bb_percent'] = (df['Close'] - lower) / (upper - lower)
    df['bb_width']   = upper - lower

    # ATR(14)
    df['atr_14'] = AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).average_true_range()

    # SMA/EMA, CCI, ADX (często spotykane)
    df['sma_20']  = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['ema_12']  = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['cci_20']  = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
    df['adx_14']  = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()

    # OBV + relacja do średniej wolumenu (wolumenowe)
    df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['vol_ma_20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio_20'] = df['Volume'] / df['vol_ma_20']

    return df


# ───────────────────────────────────────────────────────────────
# 3) CECHY ROLLING / ZASIĘGI / KALENDARZ
# ───────────────────────────────────────────────────────────────
def add_rolling_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Zasięgi i relacje cenowe
    df['high_low_range']   = (df['High'] - df['Low']) / df['Close']
    df['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open']

    # Zmienność na bazie zwrotów
    if 'ret_1d' not in df:
        df['ret_1d'] = df['Close'].pct_change(1)
    df['ret_std_5']  = df['ret_1d'].rolling(5).std()
    df['ret_std_10'] = df['ret_1d'].rolling(10).std()
    df['ret_std_20'] = df['ret_1d'].rolling(20).std()

    # Rolling średnie/odchylenia Close
    df['roll_mean_5']  = df['Close'].rolling(5).mean()
    df['roll_mean_10'] = df['Close'].rolling(10).mean()
    df['roll_mean_20'] = df['Close'].rolling(20).mean()
    df['roll_std_5']   = df['Close'].rolling(5).std()
    df['roll_std_10']  = df['Close'].rolling(10).std()
    df['roll_std_20']  = df['Close'].rolling(20).std()

    return df


def add_calendar_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # wymagane: df['Date'] jako datetime
    dow = df['Date'].dt.weekday  # 0..6
    # kodowanie cykliczne (dzień tygodnia)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    return df


# ───────────────────────────────────────────────────────────────
# 4) GŁÓWNA: 4 PRESETY CECH
#     dataset_type ∈ {"ohlc","ta_lags","mixed","ta_lags_plus"}
# ───────────────────────────────────────────────────────────────
def prepare_data(file_path: str = 'data/gold_data.csv',
                 dataset_type: str = "ta_lags",
                 test_size: float = 0.2,
                 scale_X: bool | None = None,
                 target_mode: str = "close_next"  # "close_next" albo "return_next"
                 ):
    """
    Zwraca: X_train, X_test, y_train, y_test, dates_test, feature_names, scaler (lub None)
    """
    # Wczytanie i sanity-check
    df = pd.read_csv(file_path, sep=';')
    if 'Date' not in df.columns:
        raise ValueError("Brak kolumny 'Date'.")
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"Brak wymaganej kolumny: {col}")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Target
    if target_mode == "close_next":
        df['target'] = df['Close'].shift(-1)
    elif target_mode == "return_next":
        df['ret_1d_now'] = df['Close'].pct_change(1)
        df['target'] = df['ret_1d_now'].shift(-1)
    else:
        raise ValueError("target_mode ∈ {'close_next','return_next'}")

    # Budowa cech wg presetów
    if dataset_type == "ohlc":
        X_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


    elif dataset_type == "mixed":
    # połączenie OHLC oraz rdzenia wskaźników z literatury
        tmp = add_indicators_praca(df).copy()
    # dołóż surowe kolumny z df
        tmp[['Open', 'High', 'Low']] = df[['Open', 'High', 'Low']]
        X_df = tmp[[
            'Open', 'High', 'Low',    # OHLC (bez duplikowania Close)
            'Close', 'Volume',        # kotwice
            'sma_10', 'wma_14',
            'mom_10',
            'stoch_k_14', 'stoch_d_14',
            'rsi_14', 'wr_14',
            'macd', 'macd_signal',
            'cci_20'
        ]].copy()


    elif dataset_type == "ta_research_set":
        # wskaźniki z artykułu + 'Close' i 'Volume' (przydają się m.in. dla trybu 'delta')
        tmp = add_indicators_praca(df)
        X_df = tmp[[
            'Close', 'Volume',        # kotwice
            'sma_10', 'wma_14',       # MA
            'mom_10',                 # momentum
            'stoch_k_14', 'stoch_d_14',
            'rsi_14', 'wr_14',
            'macd', 'macd_signal',
            'cci_20'
        ]].copy()

    elif dataset_type == "ta_research_set_RF":
        # wskaźniki z artykułu + 'Close' i 'Volume' (przydają się m.in. dla trybu 'delta')
        tmp = add_indicators_praca(df)
        X_df = tmp[[
            'Close', 'Volume', 'macd'
        ]].copy()

    elif dataset_type == "ta_research_set_DT":
        # wskaźniki z artykułu + 'Close' i 'Volume' (przydają się m.in. dla trybu 'delta')
        tmp = add_indicators_praca(df)
        X_df = tmp[[
            'Close', 'Volume', 'cci_20'
        ]].copy()

    elif dataset_type == "ta_research_set_XGB":
        # wskaźniki z artykułu + 'Close' i 'Volume' (przydają się m.in. dla trybu 'delta')
        tmp = add_indicators_praca(df)
        X_df = tmp[[
            'Close', 'mom_10'
        ]].copy()

    else:
        raise ValueError("dataset_type ∈ {'ohlc','ta_lags','mixed','ta_lags_plus','ta_lags_2','ta_lags_core','praca'}")

   
    
    

    # Drop NaN (po wskaźnikach/lagach) i target NaN
    df_full = pd.concat([df[['Date', 'target']], X_df], axis=1)
    df_full = df_full.dropna().reset_index(drop=True)

    dates = df_full['Date'].values
    y = df_full['target'].values
    X = df_full.drop(columns=['Date', 'target'])

    # Chronologiczny split
    n = len(df_full)
    split_idx = int(n * (1 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Niepoprawny split – za mało danych.")
    X_train_df, X_test_df = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    # Skalowanie (auto lub zgodnie z parametrem)
    if scale_X is None:
        scale_X = (dataset_type != "ohlc")  # ohlc domyślnie bez skalowania

    scaler = None
    if scale_X:
        scaler = StandardScaler().fit(X_train_df)
        X_train = scaler.transform(X_train_df)
        X_test  = scaler.transform(X_test_df)
    else:
        X_train = X_train_df.values
        X_test  = X_test_df.values

    feature_names = list(X.columns)
    print(f"[prepare_data] Preset='{dataset_type}', cech={len(feature_names)}, "
          f"target='{target_mode}', scale_X={scale_X}")
    return X_train, X_test, y_train, y_test, dates_test, feature_names, scaler