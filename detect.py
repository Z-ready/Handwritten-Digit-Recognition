from PIL import Image
import torch
import torchvision
from torch import nn
import torch.nn.functional as F


img_path = 'NUMBER IMAGE'
img = Image.open(img_path)
print(img)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),  # 将图像大小调整为 32x32
    torchvision.transforms.Grayscale(),       # 将图像转换为灰度图像
    torchvision.transforms.ToTensor()
])
image = transform(img).unsqueeze(0) 

class HandWrite(nn.Module):
    def __init__(self):
        super(HandWrite, self).__init__()
        # 第一层卷积: 1 通道 -> 32 通道, 卷积核 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二层卷积: 64 通道 -> 128 通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Dropout 层防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 28x28 -> 14x14
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # 14x14 -> 7x7
        
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 输出类别
        return x


model_path = 'first_HW_9_GPU.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
module = torch.load(model_path, map_location=device)
print(module)
module.eval()
with torch.no_grad():

    output = module(image)
    predicted_class = output.argmax(1).item()
    print(f'Predicted class: {predicted_class}')
