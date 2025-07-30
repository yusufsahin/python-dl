import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(42)
weights = np.random.rand(2, 1)
bias = np.random.rand(1)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    error = y - y_pred

    d_weights = np.dot(X.T, error * sigmoid_derivative(y_pred))
    d_bias = np.sum(error * sigmoid_derivative(y_pred))

    weights += learning_rate * d_weights
    bias += learning_rate * d_bias

print("\nXOR Tahmin Sonuçları (tek katmanla):")
for x_input in X:
    output = sigmoid(np.dot(x_input, weights) + bias)
    print(f"Girdi: {x_input} => Tahmin: {output.round(3)}")