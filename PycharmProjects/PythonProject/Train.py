import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root='/Users/vow/Documents/MNIST/', train=True, download=False, transform=transform)
test_set = torchvision.datasets.MNIST(root='/Users/vow/Documents/MNIST/', train=False, download=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# for images, labels in train_loader:
#     print(images.shape)
#     print(labels.shape)
#     print(labels[:10])
#     break

class MLP0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 512)

    def forward(self, x):
        return x

# model = MLP0()
#
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr = 0.001)
#
# images, labels = next(iter(train_loader))
# outputs = model(images)
# print(outputs.shape)

class NET(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NET()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_func = criterion

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        output = model(images)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), 'mnist_model.pth') #保存
print("模型保存为 mnist_model.pth")

model.load_state_dict(torch.load('mnist_model.pth'))

for images, labels in test_loader:
    plt.imsave('test.png', images[0].squeeze(), cmap='gray')
    print(f"保存的图片真实标签是: {labels[0].item()}")
    break



