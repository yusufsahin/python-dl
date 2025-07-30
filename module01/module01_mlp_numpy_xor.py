import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(1, hidden_size)
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(1, output_size)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    error = y - y_pred
    loss = np.mean(error ** 2)

    d_output = error * sigmoid_derivative(y_pred)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

    W2 += learning_rate * np.dot(a1.T, d_output)
    b2 += learning_rate * np.sum(d_output, axis=0, keepdims=True)
    W1 += learning_rate * np.dot(X.T, d_hidden)
    b1 += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.5f}")

print("\nMLP ile XOR Tahminleri:")
for x_input in X:
    a1 = sigmoid(np.dot(x_input, W1) + b1)
    output = sigmoid(np.dot(a1, W2) + b2)
    print(f"Girdi: {x_input} => Tahmin: {output.round(3)}")
