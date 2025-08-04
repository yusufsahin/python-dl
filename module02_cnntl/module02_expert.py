import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# GPU kontrolü
if not torch.cuda.is_available():
    raise RuntimeError("CUDA destekli bir GPU bulunamadı. Eğitim için GPU gereklidir!")

# GPU'ya zorla
device = torch.device("cuda")
print(f"✅ Kullanılan cihaz: {torch.cuda.get_device_name()} (cuda)")

# 1. CIFAR-10 verisini hazırla
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet boyutu
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# 2. Pretrained ResNet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 3. Eğitim + doğrulama + grafik
train_acc_list, val_acc_list = [], []

for epoch in range(10):
    model.train()
    correct = total = 0
    for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/10"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_acc_list.append(train_acc)

    # Val
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_acc_list.append(val_acc)
    print(f"Epoch {epoch+1} - Train: {train_acc:.4f} - Val: {val_acc:.4f}")

# 4. Modeli kaydet
torch.save(model.state_dict(), "resnet18_cifar10.pt")

# 5. Grafik
plt.plot(train_acc_list, label="Eğitim")
plt.plot(val_acc_list, label="Doğrulama")
plt.title("Eğitim Grafiği")
plt.grid(), plt.legend(), plt.show()