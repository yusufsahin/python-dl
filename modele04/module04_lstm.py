import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# 🔧 HYPERPARAMETERS
SEQ_LEN = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 50
LR = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 📥 DATA
df = yf.download("AAPL", start="2010-01-01", end="2024-12-31")[['Close']]
data = df.values.reshape(-1, 1)

# 🔄 NORMALIZATION
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# 📊 CREATE SEQUENCES
def create_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

X, Y = create_sequences(data, SEQ_LEN)

# 🧪 SPLIT DATASET
split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# 🔄 TO TENSOR
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)


# 🧠 LSTM MODEL
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = StockLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# 🏋️ TRAINING
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train)
    loss = criterion(output, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")

# 📊 EVALUATION
model.eval()
with torch.no_grad():
    predicted = model(X_test).cpu().numpy()
    true = Y_test.cpu().numpy()

predicted_prices = scaler.inverse_transform(predicted)
true_prices = scaler.inverse_transform(true)

# 🎯 METRICS
rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
mae = mean_absolute_error(true_prices, predicted_prices)
r2 = r2_score(true_prices, predicted_prices)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE : {mae:.2f}")
print(f"✅ R²  : {r2:.4f}")

# 📈 PLOT
plt.figure(figsize=(12, 6))
plt.plot(true_prices, label="Gerçek")
plt.plot(predicted_prices, label="Tahmin")
plt.title("📈 AAPL Hisse Fiyat Tahmini (LSTM)")
plt.xlabel("Gün")
plt.ylabel("Fiyat ($)")
plt.legend()
plt.grid()
plt.show()
