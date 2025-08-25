import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, CCIIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands


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

    elif dataset_type == "ta_lags_2":
        # Odchudzony zestaw: rdzeń + po 1 reprezentancie z rodzin
        # - trend/momentum: macd_diff (rdzeń)
        # - pozycja/zmienność: bb_percent (rdzeń), atr_14 (reprezentant zmienności)
        # - lag/zwrot: close_lag_1, ret_5d (po 1 szt.)
        # - wolumen: Volume (najprostszy reprezentant)
        tmp = add_core_indicators(df)
        tmp = add_basic_lags(tmp, lag_list=(1,), add_returns=True)
        X_df = tmp[[
            'Close',           # potrzebne do rekonstrukcji przy uczeniu na delcie
            'Volume',          # 1 reprezentant wolumenu
            'macd_diff',       # rdzeń (trend/momentum)
            'bb_percent',      # rdzeń (pozycja względem pasm)
            'atr_14',          # 1 reprezentant zmienności
            'close_lag_1',     # 1 krótki lag
            'ret_5d',          # 1 zwrot o średnim horyzoncie
            'rsi_14'
        ]].copy()

    elif dataset_type == "ta_lags_core":
        # Minimalny rdzeń do testu: tylko cechy, które naprawdę „niosą” wynik
        # + technicznie niezbędne kolumny (Close, ewentualnie atr_14).
        tmp = add_core_indicators(df)
        # lags/returns nie są tu potrzebne; zostawiamy tylko rdzeń + Close (+ ATR)
        # jeśli chcesz, możesz dodać lag_1, ale intencją jest maksymalny minimalizm.
        X_df = tmp[[
            'Close',       # niezbędne do rekonstrukcji poziomu przy delcie
            'Volume',   # kluczowe w obu miarach ważności
            'atr_14',      # zostawiamy, bo bywa potrzebny dla delta_over_atr
            #'rsi_14'
        ]].copy()

    elif dataset_type == "ta_lags":
        tmp = add_core_indicators(df)
        tmp = add_basic_lags(tmp, lag_list=(1, 5, 10), add_returns=True)
        X_df = tmp[[
            'Close', 'Volume',
            'macd_diff', 'rsi_14', 'bb_percent', 'bb_width', 'atr_14',
            'sma_20', 'ema_12', 'cci_20', 'adx_14',
            'close_lag_1', 'close_lag_5', 'close_lag_10',
            'ret_1d', 'ret_5d'
        ]].copy()

    elif dataset_type == "mixed":
        tmp = add_core_indicators(df)
        tmp = add_basic_lags(tmp, lag_list=(1, 5, 10), add_returns=True)
        X_df = tmp[[
            'Open', 'High', 'Low', 'Close', 'Volume',
            'macd_diff', 'rsi_14', 'bb_percent', 'bb_width', 'atr_14',
            'sma_20', 'ema_12', 'cci_20', 'adx_14',
            'close_lag_1', 'close_lag_5', 'close_lag_10',
            'ret_1d', 'ret_5d'
        ]].copy()

    elif dataset_type == "ta_lags_plus":
        # bogatszy preset pod nieliniowe modele (i test na drzewach)
        tmp = add_core_indicators(df)
        tmp = add_basic_lags(tmp, lag_list=(1, 2, 3, 5, 10, 20), add_returns=True)
        tmp = add_rolling_blocks(tmp)
        tmp = add_calendar_cyclical(tmp)

        # selekcja sensownej puli (unikamy setek kolumn)
        X_df = tmp[[
            # kotwice cenowe i wolumenowe
            'Close', 'Volume',

            # rdzeń TA
            'macd_diff', 'rsi_14', 'bb_percent', 'bb_width', 'atr_14',
            'sma_20', 'ema_12', 'cci_20', 'adx_14',
            'obv', 'vol_ratio_20',

            # lagi i zwroty
            'close_lag_1', 'close_lag_2', 'close_lag_3',
            'close_lag_5', 'close_lag_10', 'close_lag_20',
            'ret_1d', 'ret_5d', 'ret_10d',

            # rolling stat / zmienność / zasięgi
            'ret_std_5', 'ret_std_10', 'ret_std_20',
            'roll_mean_5', 'roll_mean_10', 'roll_mean_20',
            'roll_std_5', 'roll_std_10', 'roll_std_20',
            'high_low_range', 'close_open_ratio',

            # kalendarz (cykliczne)
            'dow_sin', 'dow_cos',
        ]].copy()
    else:
        raise ValueError("dataset_type ∈ {'ohlc','ta_lags','mixed','ta_lags_plus'}")
    
    

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