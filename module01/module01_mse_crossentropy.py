import numpy as np
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.3])


mse = np.mean((y_true - y_pred) ** 2)
def binary_cross_entropy(y, yhat):
    epsilon = 1e-10
    yhat = np.clip(yhat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

bce = binary_cross_entropy(y_true, y_pred)

print("MSE:", round(mse, 4))
print("Binary Cross Entropy:", round(bce, 4))



