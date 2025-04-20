import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_lstm_data(file_path='data/gold_data.csv', window_size=10, test_size=0.2):
    print("[INFO] Wczytywanie danych...")
    df = pd.read_csv(file_path, sep=';')
    print("[INFO] Dane:")
    print(df.head())
    print("[INFO] Kolumny:", df.columns.tolist())

    if 'Close' not in df.columns:
        print("[ERROR] Brak kolumny 'Close'.")
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    features = ['Open', 'High', 'Low', 'Volume']
    target = 'Close'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    try:
        X_scaled = scaler_x.fit_transform(df[features])
        y_scaled = scaler_y.fit_transform(df[[target]])
    except Exception as e:
        print("[ERROR] Błąd podczas skalowania:", e)
        return None

    X_seq, y_seq, dates_seq = [], [], []
    for i in range(len(df) - window_size):
        X_seq.append(X_scaled[i:i + window_size])
        y_seq.append(y_scaled[i + window_size])
        dates_seq.append(df['Date'].iloc[i + window_size])

    print(f"[INFO] Liczba sekwencji: {len(X_seq)}")

    if len(X_seq) == 0:
        print("[ERROR] Brak wygenerowanych sekwencji!")
        return None

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    dates_seq = np.array(dates_seq)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X_seq, y_seq, dates_seq, test_size=test_size, shuffle=False
    )

    print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, dates_test, scaler_y