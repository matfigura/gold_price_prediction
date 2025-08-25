import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ——————————————————————————————————————————
# 0) Ustawienia losowości (reproducibility)
# ——————————————————————————————————————————
def _set_seeds(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    # (opcjonalnie) można włączyć determinism, ale nie na każdej maszynie jest wspierany
    # try:
    #     tf.config.experimental.enable_op_determinism(True)
    # except Exception:
    #     pass

# ——————————————————————————————————————————
# 1) Budowa i kompilacja modelu
# ——————————————————————————————————————————
def build_lstm_model(
    input_shape,
    units_1: int = 128,
    units_2: int = 32,
    dropout_rate: float = 0.2,
    lr: float = 1e-3
) -> tf.keras.Model:
    """
    Prosty, skuteczny model sekwencyjny:
      LSTM(units_1, return_sequences=True) -> Dropout
      LSTM(units_2, return_sequences=False) -> Dropout
      Dense(1)
    """
    _set_seeds(42)

    model = Sequential([
        LSTM(units_1, return_sequences=True, activation='tanh', input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units_2, return_sequences=False, activation='tanh'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mae')
    return model

# ——————————————————————————————————————————
# 2) Trening z walidacją chrono + EarlyStopping
# ——————————————————————————————————————————
def train_lstm(
    X_train, y_train,
    val_fraction: float = 0.1,
    epochs: int = 60,
    batch_size: int = 64,
    units_1: int = 128,
    units_2: int = 32,
    dropout_rate: float = 0.2,
    lr: float = 1e-3,
    es_patience: int = 8,
    rlrop_patience: int = 4,
    verbose: int = 2
):
    """
    Uczy model LSTM bez tasowania, z walidacją z końcówki TRAIN.
    Zwraca: (model, history, (X_val, y_val)) dla ewentualnej diagnostyki.
    """
    # — split chrono: walidacja = ostatnie val_fraction TRAIN
    n = len(X_train)
    val_n = max(1, int(val_fraction * n))
    X_tr, y_tr = X_train[:-val_n], y_train[:-val_n]
    X_val, y_val = X_train[-val_n:], y_train[-val_n:]

    # — dane jako tf.data (szybciej/pewniej)
    ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # — model + callbacki
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, units_1=units_1, units_2=units_2, dropout_rate=dropout_rate, lr=lr)

    early_stop = EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=rlrop_patience, min_lr=1e-5)

    # — logi
    print("[LSTM] ▶ Start treningu")
    print(f"[LSTM]   TRAIN: {X_tr.shape}  VAL: {X_val.shape}  batch_size={batch_size}")
    t0 = time.time()

    history = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose,
        shuffle=False  # ← ważne przy sekwencjach
    )

    t1 = time.time()
    print(f"[LSTM] ✅ Zakończono trening w {t1 - t0:.2f}s (best val_loss={min(history.history['val_loss']):.6f})")

    return model, history, (X_val, y_val)

# ——————————————————————————————————————————
# 3) Predykcja
# ——————————————————————————————————————————
def predict_lstm(model: tf.keras.Model, X_test: np.ndarray) -> np.ndarray:
    """
    Zwraca płaską tablicę predykcji.
    """
    ds_test = tf.data.Dataset.from_tensor_slices(X_test).batch(256).prefetch(tf.data.AUTOTUNE)
    return model.predict(ds_test, verbose=0).ravel()