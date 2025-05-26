
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(file_path='data/gold_data.csv'):
    df = pd.read_csv(file_path, sep=';')

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')

    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=['Date', 'Close'], errors='ignore')
    y = df['Close']
    dates = df['Date']
    features = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X_scaled, y, dates, test_size=0.2, random_state=42
    )

    print("Dane przygotowane:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test, dates_test, features