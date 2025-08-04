# TensorFlow ve gerekli modülleri yükle
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# 1. Veri setini yükle ve normalize et (0-255 → 0-1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0  # (28,28) → (28,28,1)

# 2. Sıralı (Sequential) model tanımı
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # Kenarları öğrenir
    layers.MaxPooling2D(),  # Özellik haritalarını küçültür
    layers.Flatten(),       # 2D → 1D vektöre dönüştürür
    layers.Dense(64, activation='relu'),  # Tam bağlı katman
    layers.Dense(10, activation='softmax')  # 10 sınıf için çıkış
])

# 3. Derleme – optimizasyon, kayıp ve metrik tanımı
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Eğitim başlat (5 epoch)
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# 5. Eğitim sonuçlarını görselleştir
#  Doğruluk grafiği
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.legend(), plt.grid(), plt.title("Doğruluk Grafiği"), plt.show()
