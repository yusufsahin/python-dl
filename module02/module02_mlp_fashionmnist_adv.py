import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ===========================
# 🔧 HİPERPARAMETRELER
# ===========================
BATCH_SIZE = 64              # Her iterasyonda kullanılacak veri adedi
EPOCHS = 50                  # Toplam eğitim turu (epoch) sayısı
LEARNING_RATE = 0.001        # Öğrenme oranı (learning rate)
USE_DROPOUT = True           # Dropout katmanı kullanılsın mı?
USE_L2 = True                # L2 regularizasyonu kullanılacak mı?
USE_BATCHNORM = True         # BatchNorm katmanı kullanılacak mı?
OPTIMIZER_TYPE = "Adam"      # Hangi optimizasyon algoritması kullanılacak?
WEIGHT_INIT = "He"           # Ağın ağırlıkları nasıl başlatılacak?
USE_EARLYSTOP = True         # Erken durdurma (early stopping) aktif mi?
EARLYSTOP_PATIENCE = 3       # Erken durdurmada sabredilecek epoch sayısı

# ===========================
# 📥 VERİ HAZIRLAMA
# ===========================
# Girdi verisini normalize et ve tensöre çevir
transform = transforms.Compose([
    transforms.ToTensor(),                         # Görüntüyü tensöre çevir (0-1)
    transforms.Normalize((0.5,), (0.5,)) # -1 ile 1 arasına normalize et
])
# FashionMNIST veri setini indir ve dönüştür
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
# Veriyi DataLoader ile batch'lere böl
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# ===========================
# 🧠 MODEL TANIMI
# ===========================
class AdvancedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Flatten()]                              # 28x28 resmi tek boyuta indir (784,)
        layers.append(nn.Linear(28 * 28, 256))               # İlk tam bağlı katman
        if USE_BATCHNORM:
            layers.append(nn.BatchNorm1d(256))               # Batch Normalization uygula
        layers.append(nn.ReLU())                             # Aktivasyon fonksiyonu
        if USE_DROPOUT:
            layers.append(nn.Dropout(0.5))                   # Dropout uygula (%50)
        layers.append(nn.Linear(256, 128))                   # İkinci katman
        if USE_BATCHNORM:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 10))                    # Çıkış katmanı (10 sınıf)
        self.model = nn.Sequential(*layers)                  # Katmanları sırayla bağla

    def forward(self, x):
        return self.model(x)                                 # İleri geçiş (forward) fonksiyonu

model = AdvancedMLP()                                        # Modeli oluştur
# ===========================
# 🧬 AĞIRLIK BAŞLATMA
# ===========================
def init_weights(m):
    if isinstance(m, nn.Linear):
        if WEIGHT_INIT == "Xavier":
            nn.init.xavier_uniform_(m.weight)                # Xavier başlatma
        elif WEIGHT_INIT == "He":
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He başlatma
        nn.init.zeros_(m.bias)                               # Bias'ı sıfırla

model.apply(init_weights)                                    # Tüm katmanlara uygula
# ===========================
# ⚙️ KAYIP FONKSİYONU & OPTİMİZER
# ===========================
criterion = nn.CrossEntropyLoss()                            # Çoklu sınıflar için uygun kayıp fonksiyonu
weight_decay = 0.01 if USE_L2 else 0.0                      # L2 regularizasyonu (weight decay) parametresi

# Seçilen optimizere göre oluştur
if OPTIMIZER_TYPE == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=weight_decay)
elif OPTIMIZER_TYPE == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
else:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

# ===========================
# 🧪 EĞİTİM DÖNGÜSÜ
# ===========================
train_losses, test_losses = [], []        # Kayıp değerlerini saklamak için listeler
best_val_loss = float('inf')              # Erken durdurma için en iyi doğrulama kaybı
patience_counter = 0                      # Sabır sayacı (erken durdurma)

for epoch in range(EPOCHS):
    model.train()                         # Modeli eğitim moduna al
    total_train_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()             # Gradientleri sıfırla
        outputs = model(images)           # Tahminleri al
        loss = criterion(outputs, labels) # Kayıp hesapla
        loss.backward()                   # Gradientleri hesapla (geri yayılım)
        optimizer.step()                  # Ağırlıkları güncelle
        total_train_loss += loss.item()   # Toplam eğitim kaybına ekle

    avg_train_loss = total_train_loss / len(train_loader)    # Ortalama eğitim kaybı
    train_losses.append(avg_train_loss)

    model.eval()                          # Modeli değerlendirme moduna al
    total_test_loss = 0
    correct = 0

    with torch.no_grad():                 # Gradient hesaplama kapalı
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            preds = outputs.argmax(dim=1)             # En yüksek olasılıklı sınıfı seç
            correct += (preds == labels).sum().item() # Doğru tahmin sayısını topla

    avg_test_loss = total_test_loss / len(test_loader)      # Ortalama doğrulama kaybı
    test_losses.append(avg_test_loss)
    acc = correct / len(test_loader.dataset)                # Doğruluk (accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_test_loss:.4f} - Acc: {acc:.4f}")

    # Erken Durdurma (Early Stopping) kontrolü
    if USE_EARLYSTOP:
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLYSTOP_PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
# ===========================
# 📉 KAYIP GRAFİKLERİ
# ===========================
plt.plot(train_losses, label='Train Loss')           # Eğitim kaybı grafiği
plt.plot(test_losses, label='Validation Loss')       # Doğrulama kaybı grafiği
plt.xlabel('Epoch')                                 # X ekseni: Epoch
plt.ylabel('Loss')                                  # Y ekseni: Kayıp
plt.title('Eğitim vs Doğrulama Kaybı')              # Grafik başlığı
plt.legend()                                        # Açıklamalar
plt.grid(True)                                      # Izgara
plt.show()                                          # Göster

# ===========================
# 📊 CONFUSION MATRIX (KARIŞIKLIK MATRİSİ)
# ===========================
all_preds = []
all_labels = []

model.eval()                                        # Modeli değerlendirme moduna al
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())              # Tahmin edilen sınıfları topla
        all_labels.extend(labels.numpy())            # Gerçek sınıfları topla

cm = confusion_matrix(all_labels, all_preds)         # Confusion matrix hesapla
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Görselleştir
plt.xlabel("Tahmin")                                # X ekseni etiketi
plt.ylabel("Gerçek")                                # Y ekseni etiketi
plt.title("Confusion Matrix")                       # Başlık
plt.show()

# ===========================
# 💾 MODELİ KAYDETME (Opsiyonel)
# ===========================
# torch.save(model.state_dict(), "advanced_mlp.pt")  # Modelin ağırlıklarını kaydet
# print("Model kaydedildi: advanced_mlp.pt")