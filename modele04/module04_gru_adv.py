import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 📥 Veri indir
df = yf.download("AAPL", start="2010-01-01", end="2024-12-31")[["Close"]]
df.dropna(inplace=True)


# 🔄 Normalize et
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 🧩 Sekans oluştur
SEQ_LEN = 60
X, Y = [], []
for i in range(SEQ_LEN, len(scaled_data)):
    X.append(scaled_data[i-SEQ_LEN:i])
    Y.append(scaled_data[i])
X, Y = np.array(X), np.array(Y)

# 🔀 Eğitim/Test ayrımı
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# 🧠 GRU modeli
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 🏋️ Eğitim
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)


# 🔍 Tahmin
preds = model.predict(X_test)
preds_inverse = scaler.inverse_transform(preds)
Y_test_inverse = scaler.inverse_transform(Y_test)

# 📊 Skorlar
rmse = np.sqrt(mean_squared_error(Y_test_inverse, preds_inverse))
mae = mean_absolute_error(Y_test_inverse, preds_inverse)
r2 = r2_score(Y_test_inverse, preds_inverse)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE : {mae:.2f}")
print(f"✅ R²  : {r2:.4f}")

# 📈 Grafik
plt.figure(figsize=(14,6))
plt.plot(Y_test_inverse, label='Gerçek')
plt.plot(preds_inverse, label='Tahmin')
plt.title('Geliştirilmiş GRU – AAPL Tahmini')
plt.xlabel('Zaman')
plt.ylabel('Fiyat ($)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

