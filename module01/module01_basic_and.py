import numpy as np

# 1. Giriş ve çıkış verileri (AND kapısı)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [0],
    [1],
    [1]
])

# 2. Aktivasyon fonksiyonu ve türevi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 3. Ağırlıkların rastgele başlatılması
np.random.seed(42)  # Aynı sonuçları almak için sabit tohum
weights = np.random.rand(2, 1)
bias = np.random.rand(1)

# 4. Eğitim parametreleri
learning_rate = 0.1
epochs = 10000

# 5. Eğitim döngüsü
for epoch in range(epochs):
    # İleri yayılım (forward propagation)
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)

    # Hata (loss)
    error = y - y_pred
    loss = np.mean(error ** 2)

    # Geri yayılım (backpropagation)
    d_weights = np.dot(X.T, error * sigmoid_derivative(y_pred))
    d_bias = np.sum(error * sigmoid_derivative(y_pred))

    # Ağırlık ve bias güncelleme
    weights += learning_rate * d_weights
    bias += learning_rate * d_bias

    # Her 1000 adımda sonucu yazdır
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.5f}")

# 6. Eğitilmiş modelin tahmini
print("\nEğitim tamamlandı.")
print("AND Kapısı Tahminleri:")
for x_input in X:
    output = sigmoid(np.dot(x_input, weights) + bias)
    print(f"Girdi: {x_input} => Tahmin: {output.round(3)}")
