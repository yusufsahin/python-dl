from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# 1. CIFAR-10 veri setini yükle
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizasyon

# 2. Augmentation (data çoğaltma) – görüntüleri çevir, döndür
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15)
datagen.fit(x_train)

# 3. CNN modeli tanımla
model = Sequential([
    # Katman 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    # Katman 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    # Çıkış kısmı
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 sınıf (uçak, kedi, köpek vs.)
])

# 4. Model derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Eğitim (datagen ile augmented eğitim)
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=10, validation_data=(x_test, y_test))

# 6. Eğitim grafikleri
plt.plot(history.history['accuracy'], label="Eğitim")
plt.plot(history.history['val_accuracy'], label="Doğrulama")
plt.title("Doğruluk"), plt.legend(), plt.grid(), plt.show()