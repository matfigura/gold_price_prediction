import time
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def _set_seeds(seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)
    # Opcjonalnie (nie wszędzie działa determinism):
    # try:
    #     tf.config.experimental.enable_op_determinism(True)
    # except Exception:
    #     pass


def build_lstm_model(
    input_shape: Tuple[int, int],
    units_1: int = 128,
    units_2: int = 32,
    dropout_rate: float = 0.2,
    lr: float = 1e-3
) -> tf.keras.Model:
    """
    LSTM(units_1, return_sequences=True) -> Dropout
    LSTM(units_2, return_sequences=False) -> Dropout
    Dense(1)
    """
    _set_seeds(42)
    model = Sequential([
        LSTM(units_1, return_sequences=True, activation="tanh", input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units_2, return_sequences=False, activation="tanh"),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mae")
    return model


def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    *,
    val_fraction: float = 0.1,   # <-- zgodnie z Twoim main
    epochs: int = 60,
    batch_size: int = 64,
    units_1: int = 128,
    units_2: int = 32,
    dropout_rate: float = 0.2,
    lr: float = 1e-3,
    es_patience: int = 8,
    rlrop_patience: int = 4,
    verbose: int = 2,
    fast: bool = False           # <-- tryb szybki
):
    """
    Trening LSTM bez tasowania, walidacja to ostatnie `val_fraction` sekwencji TRENINGOWYCH.
    Zwraca: (model, history, (X_val, y_val))
    """
    # Fast-presety (krótszy i tańszy trening)
    if fast:
        epochs       = min(epochs, 25)
        batch_size   = 128 if batch_size == 64 else batch_size
        units_1      = min(units_1, 96)
        units_2      = min(units_2, 32)
        es_patience  = min(es_patience, 5)
        rlrop_patience = min(rlrop_patience, 3)

    # Chronologiczny split walidacyjny z końcówki TRAIN
    n = len(X_train)
    val_n = max(1, int(val_fraction * n))
    X_tr, y_tr = X_train[:-val_n], y_train[:-val_n]
    X_val, y_val = X_train[-val_n:], y_train[-val_n:]

    # tf.data (szybciej i stabilniej pamięciowo)
    ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Model + callbacki
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, units_1=units_1, units_2=units_2,
                             dropout_rate=dropout_rate, lr=lr)

    early_stop = EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=rlrop_patience, min_lr=1e-5)

    print("[LSTM] ▶ Start treningu")
    print(f"[LSTM]   TRAIN: {X_tr.shape}  VAL: {X_val.shape}  batch_size={batch_size}  fast={fast}")
    t0 = time.time()

    history = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose,
        shuffle=False  # ważne: bez tasowania sekwencji
    )

    t1 = time.time()
    print(f"[LSTM] ✅ Zakończono w {t1 - t0:.2f}s (best val_loss={min(history.history['val_loss']):.6f})")

    return model, history, (X_val, y_val)


def predict_lstm(model: tf.keras.Model, X_test: np.ndarray) -> np.ndarray:
    ds_test = tf.data.Dataset.from_tensor_slices(X_test).batch(256).prefetch(tf.data.AUTOTUNE)
    return model.predict(ds_test, verbose=0).ravel()