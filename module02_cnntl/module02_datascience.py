# 📦 Gerekli modülleri içe aktar
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

# GPU kontrolü
if not torch.cuda.is_available():
    raise RuntimeError("CUDA destekli bir GPU bulunamadı. Eğitim için GPU gereklidir!")

# GPU'ya zorla
device = torch.device("cuda")
print(f"✅ Kullanılan cihaz: {torch.cuda.get_device_name()} (cuda)")


# 2️⃣ CIFAR-10 verisi için transform: boyut + normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 224x224 bekler
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # R,G,B normalizasyonu
])

# 3️⃣ Eğitim ve test verisini yükle
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 4️⃣ ResNet18 modelini önceden eğitilmiş ağırlıklarla yükle (güncel API)
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 🔒 Önceden eğitilmiş katmanları dondur (sadece fc eğitilsin)
for param in model.parameters():
    param.requires_grad = False

# 🔁 Son fully connected (fc) katmanını CIFAR-10 için yeniden tanımla
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 sınıf var
model = model.to(device)

# 5️⃣ Loss fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 🎯 Eğitim ve doğrulama doğruluklarını saklamak için listeler
train_accuracies = []
val_accuracies = []

# 6️⃣ Eğitim ve doğrulama döngüsü
for epoch in range(5):
    model.train()
    correct = total = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # Grad reset
        outputs = model(images)          # Tahmin
        loss = criterion(outputs, labels)  # Kayıp hesapla
        loss.backward()                 # Geri yayılım
        optimizer.step()                # Ağırlıkları güncelle

        _, predicted = torch.max(outputs.data, 1)  # Sınıf tahmini
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_accuracies.append(train_acc)

    # 🔍 Doğrulama
    model.eval()
    correct = total = 0
    with torch.no_grad():  # Grad hesaplama kapalı
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_accuracies.append(val_acc)

    # 📢 Sonuçları yazdır
    print(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

# 7️⃣ Grafikle doğrulukları görselleştir
plt.plot(train_accuracies, label='Eğitim Doğruluğu')
plt.plot(val_accuracies, label='Doğrulama Doğruluğu')
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.title("ResNet18 Transfer Learning (CIFAR-10)")
plt.grid()
plt.legend()
plt.show()
