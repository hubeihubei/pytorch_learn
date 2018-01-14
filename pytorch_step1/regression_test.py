import datetime

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt

# title:线性回归

tensor = torch.linspace(-1, 1, 100)
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
#x.size（[100,1]）
x = torch.unsqueeze(tensor, dim=1)

# torch.rand(size)产生size个随机数（0-1之间）也可以直接产生矩阵torch.rand(5,3)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x).cuda(), Variable(y).cuda()


# 散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.scatter(x, y, color='red',marker='o')
# plt.show()

# 定义pytorch网络
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = func.relu(self.hidden(x))
        y = self.predict(x)
        return y


# 构建网络
net = Net(1, 10, 1).cuda()
print(net)
# 启动交互
plt.ion()

# 选择优化方法
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# 选择损失函数
# MSELoss 均方差
loss_func = torch.nn.MSELoss()
star = None
end = None
for t in range(1600):
    # star = datetime.datetime.now().timestamp()
    # 预测值
    prediction = net(x)
    # prediction.cuda()
    # 计算损失
    loss = loss_func(prediction, y)
    # 每次迭代清空上一次的梯度
    optimizer.zero_grad()
    # 反向传播，计算梯度
    loss.backward()
    # 以学习效率0.5 更新梯度
    optimizer.step()
    if t % 100 == 0 or t == 1599:
        # 清除轴空间
        plt.cla()
        #torch.cuda.FloatTensor(),不可以被转为numpy,因为numpy不支持GPU，所以要把torch.cuda.FloatTensor()转到cpu上
        plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy())
        plt.plot(x.cpu().data.numpy(), prediction.cpu().data.numpy(), color='red', lw=5)
        # 标记文本
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        # 暂停
        plt.pause(0.1)
        plt.show()
    # end = datetime.datetime.now().timestamp()
# print("time=", end - star)
# 关闭交互
plt.ioff()