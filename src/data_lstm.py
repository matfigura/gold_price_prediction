import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import add_technical_indicators
from src.analysis_feature import features_1, features_2, features_3


def create_lstm_data(
    file_path='data/gold_data.csv',
    window_size=10,
    test_size=0.2,
    include_ohlc=True,
    include_features=True,
    include_technical_indicators=True,
    drop_raw_ohlc_after_feature_gen=False,
    feature_set="features_3"
):
    """
    Przygotowuje dane sekwencyjne dla LSTM z zachowaniem:
    - dopasowania skalerów WYŁĄCZNIE na części treningowej (podział chrono),
    - tworzenia sekwencji (window_size),
    - braku tasowania (shuffle=False).

    Zwraca:
    X_train: ndarray (n_próbek, window_size, n_cech)
    X_test: ndarray (n_próbek, window_size, n_cech)
    y_train: ndarray (n_próbek,)
    y_test: ndarray (n_próbek,)
    dates_test: ndarray (daty odpowiadające testowym próbom)
    scaler_y: obiekt StandardScaler, do odwracania skalowania y
    feature_names: lista nazw cech (w kolejności zgodnej z trzecim wymiarem X)
    """
    # 📥 Wczytywanie danych
    df = pd.read_csv(file_path, sep=';')

    # 🗓️ Przetwarzanie kolumny Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')

    # ✅ Upewnienie się, że mamy podstawowe kolumny
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Brakuje wymaganych kolumn: {required_cols}")

    # ➕ mid_price
    df['mid_price'] = (df['High'] + df['Low']) / 2

    # ➕ Wskaźniki techniczne
    if include_technical_indicators:
        df = add_technical_indicators(df)

    # 🔧 (opcjonalnie) dodatkowe cechy na bazie OHLC
    if include_features:
        # np. df = generate_ohlc_features(df, lags=3)
        pass

    # 🎯 Target = Close przesunięty o 1 (przewidujemy jutrzejsze Close)
    df['target'] = df['Close'].shift(-1)

    # 🧹 Usunięcie NaN (po wskaźnikach i shifcie)
    df = df.dropna(subset=['target']).dropna()

    # 🧹 Rezygnacja z surowego OHLC, jeśli wybrano
    if not include_ohlc or drop_raw_ohlc_after_feature_gen:
        df = df.drop(columns=['Open', 'High', 'Low', 'Close'], errors='ignore')

    # 🧱 Budowa macierzy cech i wektorów docelowych
    to_drop = ['Date', 'target']
    X_df = df.drop(columns=to_drop, errors='ignore').copy()
    y = df['target'].values
    dates = df['Date'].values if 'Date' in df.columns else np.arange(len(df))

    # 🔖 Zapamiętanie nazw cech
    feature_names = X_df.columns.tolist()

    # 🔹 Filtr kolumn wg feature_set
    if feature_set == "features_1":
        chosen = features_1
    elif feature_set == "features_2":
        chosen = features_2
    elif feature_set == "features_3":
        chosen = features_3
    else:
        chosen = X_df.columns.tolist()   # "all"

    missing = [c for c in chosen if c not in X_df.columns]
    if missing:
        raise ValueError(f"Lack of columns {missing} in X_df for LSTM")

    X_df = X_df[chosen].copy()

    # ─────────────────────────────────────────────
    # 🔻 PODZIAŁ „PO CZASIE” + SKALOWANIE TYLKO NA TRAIN
    # ─────────────────────────────────────────────
    n = len(X_df)
    split_idx = int(n * (1 - test_size))
    if split_idx <= 0:
        raise ValueError("Za mało danych do wyznaczenia części treningowej.")
    if split_idx <= window_size:
        raise ValueError(
            f"Okno ({window_size}) jest zbyt duże względem części treningowej ({split_idx}). "
            f"Zmień window_size lub test_size."
        )

    scaler_x = StandardScaler().fit(X_df.iloc[:split_idx])                 # fit tylko na train
    scaler_y = StandardScaler().fit(y[:split_idx].reshape(-1, 1))          # fit tylko na train (y)

    X_scaled = scaler_x.transform(X_df)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    # 📊 Generowanie sekwencji dla LSTM (na CAŁOŚCI danych, ale granica dzielenia niżej)
    X_seq, y_seq, dates_seq = [], [], []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i + window_size])
        y_seq.append(y_scaled[i + window_size])
        dates_seq.append(dates[i + window_size])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    dates_seq = np.array(dates_seq)

    # 🧱 Granica między train/test w przestrzeni SEKWENCJI:
    # jeżeli output na indeksie (i + window_size) == split_idx,
    # to pierwszy indeks sekwencji dla TEST to i = split_idx - window_size
    boundary = split_idx - window_size
    if boundary <= 0:
        raise ValueError(
            f"Granica sekwencji ({boundary}) niepoprawna. "
            f"Zwiększ liczbę danych treningowych lub zmniejsz window_size."
        )

    X_train, y_train, X_test, y_test = X_seq[:boundary], y_seq[:boundary], X_seq[boundary:], y_seq[boundary:]
    dates_test = dates_seq[boundary:]

    return X_train, X_test, y_train, y_test, dates_test, scaler_y, chosen