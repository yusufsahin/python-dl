import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report


# IMDb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)


model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.2, epochs=3, batch_size=64)

# Tahmin
y_pred = (model.predict(x_test) > 0.5).astype("int32")
print("📌 Test Accuracy:", model.evaluate(x_test, y_test)[1])
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))
