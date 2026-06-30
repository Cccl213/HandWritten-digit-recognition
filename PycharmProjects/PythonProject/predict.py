import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import os   # ← 新增

script_dir = os.path.dirname(os.path.abspath(__file__))

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
model_path = os.path.join(script_dir, 'mnist_model.pth')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

def predict(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(image_path)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        pred = outputs.argmax(dim=1).item()

    return pred

# if __name__ == '__main__':
#     result = predict('test.png')
#     print(f"预测数字: {result}")

st.title("手写数字识别器")
uploaded_file = st.file_uploader("上传一张手写数字图片", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = predict("temp.png")
    st.image(uploaded_file, caption=f'预测结果: **{result}**')
