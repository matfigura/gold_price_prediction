import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import (
    MACD, EMAIndicator, SMAIndicator,
    CCIIndicator, WMAIndicator,
    ADXIndicator, IchimokuIndicator
)
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_technical_indicators(df):
    df = df.copy()

    # 1️⃣ RSI (Relative Strength Index)
    df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # 2️⃣ MACD i linia sygnału
    macd = MACD(
        close=df['Close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # 3️⃣ ATR (Average True Range)
    df['atr'] = AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    ).average_true_range()

    # 4️⃣ EMA i SMA
    df['ema_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['sma_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()

    # 5️⃣ OBV (On-Balance Volume)
    df['obv'] = OnBalanceVolumeIndicator(
        close=df['Close'],
        volume=df['Volume']
    ).on_balance_volume()

    # 6️⃣ WROBV (Windowed Relative OBV)
    window = 14
    obv_series = df['obv']
    df['obv_roll_sum'] = obv_series.rolling(window=window).apply(
        lambda x: x.iloc[-1] - x.iloc[0], raw=False
    )
    df['vol_roll_sum'] = df['Volume'].rolling(window=window).sum()
    df['wrobv'] = df['obv_roll_sum'] / df['vol_roll_sum']

    # 7️⃣ Stochastic Oscillator %K i %D
    stoch = StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # 8️⃣ CCI (Commodity Channel Index)
    df['cci'] = CCIIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=20
    ).cci()

    # 9️⃣ Bollinger Bands (wstęgi)
    bb = BollingerBands(
        close=df['Close'],
        window=20,
        window_dev=2
    )
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    # 🔟 ADX (Average Directional Index)
    df['adx'] = ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    ).adx()

    # 1️⃣1️⃣ WMA (Weighted Moving Average)
    df['wma_14'] = WMAIndicator(close=df['Close'], window=14).wma()

    # 1️⃣2️⃣ Fibonacci Retracement % w oknie 14 dni
    fib_window = 14
    df['fib_high'] = df['High'].rolling(window=fib_window).max()
    df['fib_low'] = df['Low'].rolling(window=fib_window).min()
    df['fib_pct'] = (df['fib_high'] - df['Close']) / (df['fib_high'] - df['fib_low'])

    # 1️⃣3️⃣ Ichimoku Cloud
    ichimoku = IchimokuIndicator(
        high=df['High'],
        low=df['Low'],
        window1=9,
        window2=26,
        window3=52
    )
    df['ichimoku_tenkan'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_kijun'] = ichimoku.ichimoku_base_line()
    df['ichimoku_span_a'] = ichimoku.ichimoku_a()
    df['ichimoku_span_b'] = ichimoku.ichimoku_b()
    # uwaga: IchimokuIndicator nie udostępnia metody lagging span w wersji 'ta',
    # dlatego pomijamy tę linię.

    # 1️⃣4️⃣ Usuń kolumny tymczasowe
    df.drop(
        columns=['obv_roll_sum', 'vol_roll_sum', 'fib_high', 'fib_low'],
        inplace=True
    )

    return df

def prepare_data(
    file_path='data/gold_data.csv',
    include_midprice=True,
    include_ohlc=False,                # teraz nie używamy surowego OHLC
    include_technical_indicators=True
):
    df = pd.read_csv(file_path, sep=';')
    # --- przetworzenie kolumny Date, target itd. ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['target'] = df['Close'].shift(-1)

    # Jeżeli chcemy zastąpić cały OHLC mid_price:
    if include_midprice:
        df['mid_price'] = (df['High'] + df['Low']) / 2
        # opcjonalnie: df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Dodajemy wskaźniki techniczne
    if include_technical_indicators:
        df = add_technical_indicators(df)

    # Teraz usuwamy surowe OHLC, bo i tak zastąpiliśmy je mid_price
    df = df.drop(columns=['Open', 'High', 'Low', 'Close'], errors='ignore')

    # Usuwamy ewentualne NaN-y powstałe przy przesunięciach i wskaźnikach
    df = df.dropna(subset=['target'])
    df = df.dropna()

    # Wyodrębniamy cechy X i cel y
    X = df.drop(columns=['Date', 'target'], errors='ignore')
    y = df['target']
    dates = df['Date']

    # Skalowanie i split – tak jak wcześniej
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X_scaled, y, dates, test_size=0.2, random_state=42
    )

    print("Liczba cech:", X.shape[1])
    return X_train, X_test, y_train, y_test, dates_test, X.columns.tolist()