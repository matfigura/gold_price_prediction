import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ——————————————————————————————————————————
# 1) Funkcja budująca (i kompilująca) LSTM
# ——————————————————————————————————————————
def build_lstm_model(input_shape, units_1=128, units_2=32, dropout_rate=0.2):
    """
    Buduje i kompiluje model LSTM o architekturze:
     - LSTM(units_1, return_sequences=True, activation='tanh')
     - Dropout(dropout_rate)
     - LSTM(units_2, return_sequences=False, activation='tanh')
     - Dropout(dropout_rate)
     - Dense(1)
    Zwraca nieroztrenowany model.
    """
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Sequential([
        LSTM(units_1, return_sequences=True, activation='tanh', input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units_2, return_sequences=False, activation='tanh'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model


# ——————————————————————————————————————————
# 2) Funkcja trenująca (build + fit)
# ——————————————————————————————————————————
def train_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Tworzy model wywołując build_lstm_model, a następnie trenuje
    (X_train, y_train) z walidacją (X_val, y_val) i EarlyStopping.
    Zwraca (model, history).
    """
    # Budujemy model na podstawie kształtu sekwencji:
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2
    )
    return model, history


# ——————————————————————————————————————————
# 3) Funkcja do predykcji
# ——————————————————————————————————————————
def predict_lstm(model, X_test):
    """
    Przyjmuje wytrenowany model i zwraca płaską listę predykcji na X_test.
    """
    return model.predict(X_test).flatten()
