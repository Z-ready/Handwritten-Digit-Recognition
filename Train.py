


import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
#设置数据
train_data = torchvision.datasets.MNIST('data',train = True, transform=torchvision.transforms.ToTensor(),download= True)
test_data = torchvision.datasets.MNIST('data',train = False, transform=torchvision.transforms.ToTensor(),download= True)

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
test_datasize = len(test_data)
#搭建网络模型
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

HW = HandWrite()
#gpu_code
HW = HW.cuda()
total_train_step = 0
total_test_step = 0

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
optimizer = torch.optim.SGD(HW.parameters(),lr = 0.0001)

epoch = 10

for i in range(epoch):
    print(f'--------开始第{i}轮训练--------')
    
    
    HW.train()
    for data in train_dataloader:
        imgs ,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = HW(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad
        loss.backward()
        optimizer.step()

        total_train_step +=1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step} Loss: {loss.item()}")


    HW.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = HW(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy



    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_datasize))
    
    total_test_step = total_test_step + 1

    torch.save(HW, f"first_HW_{i}.pth")
    print("模型已保存")
