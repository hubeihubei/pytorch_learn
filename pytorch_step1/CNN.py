import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from torchvision import transforms

EPOCH = 2
BATCH_SIZE = 50
LR = 0.001
# 是否下载MNIST,如果已经下了就False
DOWNLOAD_MNIST = True
# 训练集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
print(train_data.train_data.size())  # torch.Size([60000, 28, 28])
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[1], cmap='gray')
# plt.title('%i' % train_data.train_labels[1])
# plt.show()
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)(batch_size,c,h,w)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 测试集
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda() / 255
test_y = test_data.test_labels[:2000].cuda()


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        conv2 = torch.nn.Sequential()
        conv2.add_module('conv2', torch.nn.Conv2d(16, 32, 5, 1, 2))
        conv2.add_module('relu2', torch.nn.ReLU())
        conv2.add_module('pool2', torch.nn.MaxPool2d(2))
        self.conv2 = conv2
        self.out = torch.nn.Linear(32 * 7 * 7, 10)
        # softmax用于计算概率分布
        self.prediction = torch.nn.Softmax()
        
    def forward(self, x):
        # print('origin x.size:',x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        # print("x.size():",x.size())
        # x.size:torch.Size([50, 32, 7, 7]),第一个为batchsize
        # x.view()将张量展平变成(batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        prediction = self.prediction(output)
        return prediction, x


cnn = CNN().cuda()
optimizer = torch.optim.Adam(cnn.parameters(), LR)
loss_func = torch.nn.CrossEntropyLoss()
# children()会返回下一级模块的迭代器
# child = list(cnn.children())
# print(child)
# name_child = list(cnn.named_children())
# print(name_child)
# modules() 会返回模型中所有模块的迭代器
# modules = list(cnn.modules())
# named_modules()会返回网络层名称和模块迭代器
# name_modules = list(cnn.named_modules())
# print(name_modules)
# print(modules)


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # b_x Size([50,1,28,28])
        b_x = Variable(b_x).cuda()
        # print('b_x', b_x.size())
        b_y = Variable(b_y).cuda()
        out = cnn(b_x)[0]
        # print(out, out.size())
        loss = loss_func(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            test_output = cnn(test_x)[0]
            # print("test_output:",test_output)
            #将计算图纸也放到GPU上
            pre_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = sum(pre_y == test_y) / test_y.size(0)
            print('epoch:', epoch, ' train loss:', loss.data[0], ' accracy:', accuracy)

test_output = cnn(test_x[:10])[0]
pre_y = torch.max(test_output, 1)[1].data.squeeze()
print(pre_y, 'prediction number')
print(test_y[:10], 'really number')
# named_parameters()输出网络层的名字和参数的迭代器
# parameters() 输出一个网络的全部参数的迭代器
for param in cnn.named_parameters():
    print(param[0],list(param[1]))