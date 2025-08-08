import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# 🔧 Parametreler
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64

# 📥 Veri Yükleme
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)


# 🧠 Model
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(LSTM_UNITS),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 🏋️ Eğitim
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
# 📈 Eğitim Grafiği
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title('Doğruluk')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title('Kayıp')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 📊 Performans
y_pred = (model.predict(x_test) > 0.5).astype("int32")
print("\n📌 Test Accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# 🧮 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negatif", "Pozitif"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# 🔍 Rastgele Yorum Tahmini
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(text_ids):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in text_ids if i >= 3])

random_idx = random.randint(0, len(x_test) - 1)
sample_input = x_test[random_idx].reshape(1, -1)
sample_pred = model.predict(sample_input)[0][0]
label = "Pozitif" if sample_pred > 0.5 else "Negatif"

print("\n📝 Yorum:", decode_review(x_test[random_idx]))
print(f"🔮 Tahmin: {label} ({sample_pred:.2f}) — Gerçek: {'Pozitif' if y_test[random_idx]==1 else 'Negatif'}")

