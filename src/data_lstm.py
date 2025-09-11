import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import add_indicators_praca

_INDICATORS = [
    "sma_10","wma_14","mom_10",
    "stoch_k_14","stoch_d_14",
    "rsi_14","wr_14",
    "macd","macd_signal","cci_20",
]

_PRESETS = {
    "ohlc": ["Open","High","Low","Close","Volume"],
    "ta_research_set": ["Close","Volume"] + _INDICATORS,
    "ta_research_set_LSTM": ["Close","sma_10","wma_14","mom_10","macd","macd_signal","cci_20"],
    "mixed": ["Open","High","Low","Close","Volume"] + _INDICATORS,
}

def _build_features(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    cols = _PRESETS.get(dataset_type)
    if cols is None:
        raise ValueError("dataset_type ∈ {'ohlc','ta_research_set','mixed','ta_research_set_LSTM'}")
    if dataset_type == "ohlc":
        return df[cols].copy()
    tmp = add_indicators_praca(df).copy()
    if dataset_type == "mixed":
        tmp[["Open","High","Low"]] = df[["Open","High","Low"]]
    return tmp[cols].copy()

def create_lstm_data(
    file_path: str = "data/gold_data.csv",
    dataset_type: str = "ta_research_set",
    target_mode: str = "close_next",
    window_size: int = 30,
    test_size: float = 0.2,
    scale_X: bool | None = None,
    target_transform: str = "level"
):
    df = pd.read_csv(file_path, sep=";")
    for c in ["Date","Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            raise ValueError(f"Brakuje kolumny: {c}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    y_next = df["Close"].shift(-1)
    X_df = _build_features(df, dataset_type)

    work = pd.concat(
        [df[["Date"]], X_df, df[["Close"]].rename(columns={"Close":"Close_t"}), y_next.rename("Close_t1")],
        axis=1
    ).dropna().reset_index(drop=True)

    dates = work["Date"].to_numpy()
    close_t = work["Close_t"].to_numpy(dtype=float)
    close_t1 = work["Close_t1"].to_numpy(dtype=float)
    X = work.drop(columns=["Date","Close_t","Close_t1"])
    feature_names = list(X.columns)

    n = len(work)
    split_idx = int(n*(1-test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Niepoprawny split — za mało danych.")

    if scale_X is None:
        scale_X = True
    if scale_X:
        scaler_x = StandardScaler().fit(X.iloc[:split_idx])
        X_mat = scaler_x.transform(X)
    else:
        scaler_x = None
        X_mat = X.to_numpy()

    X_seq, y_raw, dates_seq, close_last_seq = [], [], [], []
    for i in range(len(X_mat) - window_size):
        j_t = i + window_size - 1
        j_t1 = i + window_size
        X_seq.append(X_mat[i:i+window_size])
        dates_seq.append(dates[j_t1])
        ct = close_t[j_t]
        ctp1 = close_t1[j_t1]
        if target_transform == "level":
            y_raw.append(ctp1)
        elif target_transform == "return":
            y_raw.append((ctp1 - ct) / (ct + 1e-8))
        else:
            raise ValueError("target_transform ∈ {'level','return'}")
        close_last_seq.append(ct)

    X_seq = np.asarray(X_seq)
    y_raw = np.asarray(y_raw)
    dates_seq = np.asarray(dates_seq)
    close_last_seq = np.asarray(close_last_seq)

    boundary = split_idx - window_size
    if boundary <= 0:
        raise ValueError(f"window_size={window_size} za duże względem części treningowej ({split_idx}).")

    X_train, X_test = X_seq[:boundary], X_seq[boundary:]
    y_train_raw, y_test_raw = y_raw[:boundary], y_raw[boundary:]
    dates_test = dates_seq[boundary:]
    close_last_test = close_last_seq[boundary:]

    scaler_y = StandardScaler().fit(y_train_raw.reshape(-1,1))
    y_train = scaler_y.transform(y_train_raw.reshape(-1,1)).ravel()
    y_test = scaler_y.transform(y_test_raw.reshape(-1,1)).ravel()

    return X_train, X_test, y_train, y_test, dates_test, scaler_y, feature_names, close_last_test
