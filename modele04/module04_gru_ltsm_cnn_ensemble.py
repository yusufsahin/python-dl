import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping

# 📥 Veri İndirme
df = yf.download("AAPL", start="2010-01-01", end="2024-12-31")[["Close"]]
data = df.values.reshape(-1, 1)

# 🔄 Normalizasyon
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ⛓️ Sekans Oluştur
SEQ_LEN = 60
def create_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

X, y = create_sequences(data_scaled, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# 🔀 Eğitim/Test Ayrımı
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 🧠 Ensemble Model: CNN + LSTM + GRU
input_layer = Input(shape=(SEQ_LEN, 1))

# CNN Bloğu
cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
cnn = MaxPooling1D(pool_size=2)(cnn)
cnn = Flatten()(cnn)


# LSTM Bloğu
lstm = LSTM(64)(input_layer)

# GRU Bloğu
gru = GRU(64)(input_layer)

# Birleştirme
combined = concatenate([cnn, lstm, gru])
dense = Dense(64, activation='relu')(combined)
dropout = Dropout(0.3)(dense)
output = Dense(1)(dropout)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')

# 🏋️ Eğitim
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 🔍 Tahmin
y_pred = model.predict(X_test)

# 🔁 Ters Normalize
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# 📊 Performans
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE : {mae:.2f}")
print(f"✅ R²  : {r2:.4f}")

# 📈 Görselleştirme
plt.figure(figsize=(14,6))
plt.plot(y_test_inv, label='Gerçek')
plt.plot(y_pred_inv, label='Tahmin')
plt.title('Ensemble (GRU + LSTM + CNN) – AAPL Tahmini')
plt.xlabel('Zaman')
plt.ylabel('Fiyat ($)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



