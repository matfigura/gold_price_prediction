import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.data_preprocessing import add_technical_indicators


def create_lstm_data(
    file_path='data/gold_data.csv',
    window_size=10,
    test_size=0.2,
    include_ohlc=True,
    include_features=True,
    include_technical_indicators=True,
    drop_raw_ohlc_after_feature_gen=False
):
    """
    Przygotowuje dane sekwencyjne dla LSTM, analogicznie do prepare_data(), z zachowaniem:
    - mid_price zamiast OHLC
    - te same wskaźniki techniczne co w add_technical_indicators
    - możliwość generacji dodatkowych cech (generate_ohlc_features)
    - tworzenie sekwencji dla LSTM (window_size)
    - nieprzypadkowy split (shuffle=False)

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

    # ➕ Dodanie mid_price analogicznie do prepare_data
    df['mid_price'] = (df['High'] + df['Low']) / 2

    # ➕ Dodanie wskaźników technicznych
    if include_technical_indicators:
        df = add_technical_indicators(df)

    # 🔧 Generacja dodatkowych cech na bazie OHLC (jeśli potrzebne)
    if include_features:
        # Tu możesz podpiąć swoją funkcję generate_ohlc_features (np. z niższej wersji)
        # Przykład (jeśli masz): df = generate_ohlc_features(df, lags=3)
        pass  # usuń pass i wstaw wywołanie generate_ohlc_features, jeśli używasz

    # 🎯 Tworzenie targetu zanim usuniemy kolumny Close
    df['target'] = df['Close'].shift(-1)

    # 🧹 Usunięcie NaN w kolumnie target i w wskaźnikach
    df = df.dropna(subset=['target']).dropna()

    # 🧹 Usunięcie surowego OHLC, jeśli wybrano drop_raw_ohlc_after_feature_gen lub include_ohlc=False
    if not include_ohlc or drop_raw_ohlc_after_feature_gen:
        df = df.drop(columns=['Open', 'High', 'Low', 'Close'], errors='ignore')

    # ❌ Usunięcie kolumn, które nie są cechami
    to_drop = ['Date', 'target']
    X_df = df.drop(columns=to_drop, errors='ignore').copy()
    y = df['target'].values
    dates = df['Date'].values if 'Date' in df.columns else np.arange(len(df))

    # 📝 Zapis nazw cech
    feature_names = X_df.columns.tolist()

    # 🔢 Skalowanie X i y
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_df)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 📊 Generowanie sekwencji dla LSTM
    X_seq = []
    y_seq = []
    dates_seq = []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i + window_size])
        y_seq.append(y_scaled[i + window_size])
        dates_seq.append(dates[i + window_size])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    dates_seq = np.array(dates_seq)

    # 🏷️ Podział na zbiór treningowy i testowy (bez mieszania)
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X_seq, y_seq, dates_seq, test_size=test_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test, dates_test, scaler_y, feature_names
