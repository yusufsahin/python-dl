import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 🔧 HYPERPARAMETERS
SEQ_LEN = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 50
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Kullanılan cihaz: {device}")

# 📥 Veri Seti
df = yf.download("AAPL", start="2010-01-01", end="2024-12-31", auto_adjust=True)[['Close']]
data = df.values.reshape(-1, 1)


# 🔄 Normalize
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# ⛓️ Sequence Oluştur
def create_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)


X, Y = create_sequences(data, SEQ_LEN)


# 🔀 Eğitim/Test Böl
split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# 🔁 Tensor'a Çevir
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # (B, T, 1)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)


# 🧠 CNN + LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)      # (B, 1, T) → Conv1D
        x = self.relu(self.conv1d(x))  # (B, 32, T)
        x = x.permute(0, 2, 1)      # (B, T, 32) → LSTM
        out, _ = self.lstm(x)       # (B, T, H)
        return self.fc(out[:, -1, :])  # (B, 1)

model = CNN_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 🏋️ Eğitim
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train)
    loss = criterion(output, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# 📊 Tahmin
model.eval()
with torch.no_grad():
    predicted = model(X_test).cpu().numpy()
    true = Y_test.cpu().numpy()

predicted_prices = scaler.inverse_transform(predicted)
true_prices = scaler.inverse_transform(true)

# 📏 Metrikler
rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
mae = mean_absolute_error(true_prices, predicted_prices)
r2 = r2_score(true_prices, predicted_prices)

print(f"\n📊 Performans:")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE : {mae:.2f}")
print(f"✅ R²  : {r2:.4f}")

# 📈 Grafik
plt.figure(figsize=(12, 6))
plt.plot(true_prices, label="Gerçek")
plt.plot(predicted_prices, label="Tahmin")
plt.title("AAPL Fiyat Tahmini (CNN + LSTM)")
plt.xlabel("Gün")
plt.ylabel("Fiyat ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
