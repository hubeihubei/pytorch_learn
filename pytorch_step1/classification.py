import datetime

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt

# title：分类
n_data = torch.ones(100, 2)
x1 = torch.normal(2 * n_data, 1)
x2 = torch.normal(-2 * n_data, 1)
x3 = torch.normal(6 * n_data, 1)
y1 = torch.zeros(100)
y2 = torch.ones(100)
y3 = torch.ones(100) * 2
print(y3.size())
x = torch.cat((x1, x2, x3), 0).type(torch.FloatTensor)
y = torch.cat((y1, y2, y3), 0).type(torch.LongTensor)
x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, marker='o', cmap='RdYlGn',
#             lw='0')
# plt.show()


# 定义pytorch网络
# method1
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = func.relu(self.hidden(x))
        y = self.predict(x)
        return y


# method2 快速搭建等同method1
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 3)
)

# 构建网络
net = Net(2, 10, 3)
print(net)
# #启动交互
plt.ion()
#
# 选择优化方法
# optimizer = torch.optim.SGD(net.parameters(), lr=0.04)
optimizer = torch.optim.SGD(net2.parameters(), lr=0.04)
# 选择损失函数
# cross-entropy loss用于度量两个概率分布之间的相似性
loss_func = torch.nn.CrossEntropyLoss()
# star = None
# end = None
for t in range(1001):
    # star = datetime.datetime.now().timestamp()
    # 预测值
    # out = net(x)
    out = net2(x)
    # out.cuda()
    # 计算损失
    loss = loss_func(out, y)
    # 每次迭代清空上一次的梯度
    optimizer.zero_grad()
    # 反向传播，计算梯度
    loss.backward()
    # 以学习效率衰减 更新梯度
    optimizer.step()
    if t % 100 == 0 or t in [3, 6]:
        # 清除轴空间
        plt.cla()
        # a为最大值，prediction为索引
        # torch.max() 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引,将dim维设定为1，其它与输入形状保持一致。
        # softmax用于计算概率分布
        a, prediction = torch.max(func.softmax(out), 1)
        # numpy.squeeze()将输入张量形状中的1 去除并返回
        pred_y = prediction.data.numpy().squeeze()
        targ_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, marker='o', cmap='RdYlGn',
                    lw='0')
        accuracy = sum(pred_y == targ_y) / 300
        # #标记文本
        plt.text(1.5, -4, 'Loss=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})

        # #暂停
        plt.pause(0.1)
        plt.show()
#     # end = datetime.datetime.now().timestamp()
# # print("time=", end - star)
# #关闭交互
plt.ioff()
