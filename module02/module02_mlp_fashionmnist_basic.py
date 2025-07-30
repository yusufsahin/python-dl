import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 🌟 HYPERPARAMETRELER
BATCH_SIZE = 64          # Her iterasyonda işlenecek örnek sayısı
EPOCHS = 20              # Eğitim setinin modelden kaç kez geçirileceği
LEARNING_RATE = 0.001    # Öğrenme oranı (çok büyükse model kararsız olur)
USE_L2 = True            # L2 regularization (weight decay) aç/kapa
USE_DROPOUT = True       # Dropout kullanılsın mı?

# 🧹 VERİ DÖNÜŞÜMÜ (Normalizasyon yapılır: ortalama=0, std=1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 📥 EĞİTİM ve TEST VERİLERİNİ YÜKLE (FashionMNIST)
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# 🧠 MLP MODELİ TANIMI (2 gizli katman, dropout dahil)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                        # 28x28 → 784 düz vektör
            nn.Linear(28 * 28, 256),             # Giriş → 256 nöron
            nn.ReLU(),                           # Aktivasyon fonksiyonu
            nn.Dropout(0.5) if USE_DROPOUT else nn.Identity(),  # Rastgele %50 nöron devre dışı
            nn.Linear(256, 128),                 # 2. gizli katman
            nn.ReLU(),                           # Aktivasyon
            nn.Linear(128, 10)                   # Çıkış (10 sınıf)
        )

    def forward(self, x):
        return self.model(x)

model = MLP()

# ⚙️ KAYIP FONKSİYONU VE OPTİMİZER
criterion = nn.CrossEntropyLoss()  # Çok sınıflı sınıflandırma için uygun
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01 if USE_L2 else 0.0  # L2 regularization aktifse cezalandırma uygulanır
)
# 📈 EĞİTİM DÖNGÜSÜ
train_losses, test_losses = [], []

for epoch in range(EPOCHS):  # 🌍 Epoch = tüm eğitim verisinin modelden 1 kez geçmesi
    model.train()  # Eğitim moduna geç (dropout aktif olur)
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()             # Gradient’leri sıfırla
        outputs = model(images)           # Tahmin yap (forward pass)
        loss = criterion(outputs, labels) # Kayıp hesapla
        loss.backward()                   # Gradient hesapla (backpropagation)
        optimizer.step()                  # Ağırlıkları güncelle
        total_loss += loss.item()         # Kayıp toplanır

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 🔎 TEST AŞAMASI (dropout devre dışı)
    model.eval()
    total_test_loss = 0
    correct = 0

    with torch.no_grad():  # Test sırasında gradient hesaplama devre dışı
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            preds = outputs.argmax(dim=1)              # En yüksek skora sahip sınıf
            correct += (preds == labels).sum().item()  # Doğru tahminleri say

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    acc = correct / len(test_loader.dataset)

    # 🔊 Performans çıktısı
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_test_loss:.4f} - Acc: {acc:.4f}")
# 📊 KAYIP GRAFİĞİ (Overfit kontrolü)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Eğitim vs Doğrulama Kaybı')
plt.grid(True)
plt.show()