import datetime

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt

# title:模型的保存和载入
tensor = torch.linspace(-1, 1, 100)
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# 返回一个新的张量，对输入的制定位置插入维度 1
x = torch.unsqueeze(tensor, dim=1)
print(x.size())
# torch.rand(size)产生size个随机数（0-1之间）也可以直接产生矩阵torch.rand(5,3)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)


# 散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.scatter(x, y, color='red',marker='o')
# plt.show()

def save():
    # 构建网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    # 选择优化方法
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    # 选择损失函数
    # MSELoss 均方差
    loss_func = torch.nn.MSELoss()
    star = None
    end = None
    for t in range(1600):
        # star = datetime.datetime.now().timestamp()
        # 预测值
        prediction = net1(x)
        # prediction.cuda()
        # 计算损失
        loss = loss_func(prediction, y)
        # 每次迭代清空上一次的梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 以学习效率0.5 更新梯度
        optimizer.step()
    #
    prediction = net1(x)
    plt.figure(1, figsize=(10, 3))
    # 绘制多个子图plt.subplot(131)表示有一行三列，当前图片为第一个
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), color='red', lw=5)

    # 保存模型
    torch.save(net1, '../model/net.pkl')
    # 不保存计算图，只保存参数，速度相对较快
    torch.save(net1.state_dict(), '../model/net_params.pkl')


def restore_net():
    # 载入模型
    net2 = torch.load('../model/net.pkl')
    prediction = net2(x)
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), color='red', lw=5)


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('../model/net_params.pkl'))
    prediction = net3(x)
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), color='red', lw=5)
    plt.show()


save()
restore_net()
restore_params()
