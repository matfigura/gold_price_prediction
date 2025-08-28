# data_lstm.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# tylko to jest potrzebne do presetów bez lagów
from src.data_preprocessing import add_indicators_praca


def _build_features(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Zwraca X_df dla LSTM — warianty BEZ lagów (presety: 'ohlc' | 'ta_research_set' | 'mixed')."""
    tmp = df.copy()

    if dataset_type == "ohlc":
        X_df = tmp[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    elif dataset_type == "ta_research_set":
        # rdzeń wskaźników z artykułu + kotwice (Close, Volume) — BEZ LAGÓW
        tmp = add_indicators_praca(tmp)
        X_df = tmp[[
            'Close', 'Volume',
            'sma_10', 'wma_14',
            'mom_10',
            'stoch_k_14', 'stoch_d_14',
            'rsi_14', 'wr_14',
            'macd', 'macd_signal',
            'cci_20'
        ]].copy()

    elif dataset_type == "ta_research_set_LSTM":
        # rdzeń wskaźników z artykułu + kotwice (Close, Volume) — BEZ LAGÓW
        tmp = add_indicators_praca(tmp)
        X_df = tmp[[
            'Close', 
             'stoch_k_14',
             'rsi_14', 'wr_14',
             'macd_signal', 'macd',
        ]].copy()

    elif dataset_type == "mixed":
        # OHLC + rdzeń wskaźników — BEZ LAGÓW
        tmp = add_indicators_praca(tmp)
        tmp[['Open', 'High', 'Low']] = df[['Open', 'High', 'Low']]
        X_df = tmp[[
            'Open', 'High', 'Low',
            'Close', 'Volume',
            'sma_10', 'wma_14',
            'mom_10',
            'stoch_k_14', 'stoch_d_14',
            'rsi_14', 'wr_14',
            'macd', 'macd_signal',
            'cci_20'
        ]].copy()

    else:
        raise ValueError("dataset_type ∈ {'ohlc','ta_research_set','mixed','ta_research_set_LSTM'}")

    return X_df


def create_lstm_data(
    file_path: str = 'data/gold_data.csv',
    dataset_type: str = 'ta_research_set',   # 'ohlc' | 'ta_research_set' | 'mixed'
    target_mode: str = 'close_next',
    window_size: int = 30,
    test_size: float = 0.2,
    scale_X: bool | None = None,             # None => auto(True)
    target_transform: str = "level",         # NEW: "level" | "delta"
):
    """
    Zwraca: X_train, X_test, y_train_scaled, y_test_scaled,
            dates_test, scaler_y, feature_names, close_last_test
    target_transform:
      - "level" : y = Close_{t+1}
      - "delta" : y = Close_{t+1} - Close_{t}, gdzie Close_{t} to ostatni Close w oknie sekwencji
    """
    import pandas as pd, numpy as np
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(file_path, sep=';')
    for col in ['Date','Open','High','Low','Close','Volume']:
        if col not in df.columns:
            raise ValueError(f"Brakuje kolumny: {col}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    if target_mode == 'close_next':
        df['target'] = df['Close'].shift(-1)
    elif target_mode == 'return_next':
        df['ret_1d_now'] = df['Close'].pct_change(1)
        df['target'] = df['ret_1d_now'].shift(-1)
    else:
        raise ValueError("target_mode ∈ {'close_next','return_next'}")

    X_df = _build_features(df, dataset_type)

# 4) Składanie ramki: DODAJEMY Close jako *oddzielną* kolumnę referencyjną
    work = pd.concat(
        [df[['Date', 'target']], X_df, df[['Close']].rename(columns={'Close': 'Close_ref'})],
        axis=1
    ).dropna().reset_index(drop=True)

    dates         = work['Date'].values
    y_level_all   = work['target'].to_numpy()
    close_ref_all = work['Close_ref'].to_numpy()     # JEDNOWYMIAROWE
    X             = work.drop(columns=['Date', 'target', 'Close_ref'])
    feature_names = list(X.columns)

    n = len(work)
    split_idx = int(n*(1-test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Niepoprawny split — sprawdź liczność danych.")

    if scale_X is None:
        scale_X = True
    if scale_X:
        scaler_x = StandardScaler().fit(X.iloc[:split_idx])
        X_scaled = scaler_x.transform(X)
    else:
        scaler_x = None
        X_scaled = X.values

    X_seq, y_seq_raw, dates_seq, close_last_seq = [], [], [], []
    for i in range(len(X) - window_size):
        j_last = i + window_size - 1     # indeks t
        j_tgt  = i + window_size         # indeks t+1

        X_seq.append(X.iloc[i:i+window_size].to_numpy() if hasattr(X, "iloc") else X[i:i+window_size])
        dates_seq.append(dates[j_tgt])

        close_t = float(close_ref_all[j_last])       # Close_t (skalowany później NIE jest)
        y_lvl   = float(y_level_all[j_tgt])          # Close_{t+1}

        if target_transform == "level":
            y_raw = y_lvl
        elif target_transform == "delta":
            y_raw = y_lvl - close_t
        else:
            raise ValueError("target_transform ∈ {'level','delta'}")

        y_seq_raw.append(y_raw)
        close_last_seq.append(close_t)

    X_seq = np.array(X_seq)
    y_seq_raw = np.array(y_seq_raw)
    dates_seq = np.array(dates_seq)
    close_last_seq = np.array(close_last_seq)

    boundary = split_idx - window_size
    if boundary <= 0:
        raise ValueError(
            f"Okno ({window_size}) zbyt duże względem części treningowej ({split_idx}). "
            f"Zmień window_size lub test_size."
        )

    X_train, X_test = X_seq[:boundary], X_seq[boundary:]
    y_train_raw, y_test_raw = y_seq_raw[:boundary], y_seq_raw[boundary:]
    dates_test = dates_seq[boundary:]
    close_last_test = close_last_seq[boundary:]

    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1,1))
    y_train = scaler_y.transform(y_train_raw.reshape(-1,1)).ravel()
    y_test  = scaler_y.transform(y_test_raw.reshape(-1,1)).ravel()

    return X_train, X_test, y_train, y_test, dates_test, scaler_y, feature_names, close_last_test

