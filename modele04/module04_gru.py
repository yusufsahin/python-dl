import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 📥 Veri İndirme
df = yf.download("AAPL", start="2010-01-01", end="2024-12-31")[["Close"]]
data = df.values.reshape(-1, 1)

# 🔄 Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ⛓️ Sequence Oluştur
SEQ_LEN = 60
def create_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

X, y = create_sequences(data_scaled, SEQ_LEN)

# 🧪 Eğitim/Test Ayrımı
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 🧠 GRU Model
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ⏳ Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 🏋️ Model Eğitimi
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 🔍 Tahmin
y_pred = model.predict(X_test)

# 🔁 Ters normalize
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)


# 🎯 Metrikler
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE : {mae:.2f}")
print(f"✅ R²  : {r2:.4f}")


# 📈 Grafik
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Gerçek")
plt.plot(y_pred_inv, label="Tahmin")
plt.title("GRU Modeli – AAPL Tahmini")
plt.xlabel("Zaman")
plt.ylabel("Fiyat ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()