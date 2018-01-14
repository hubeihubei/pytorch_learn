import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 过拟合常发生再数据量小，网络构造复杂或庞大
# dropout 随机屏蔽一部分节点，防止过度依赖某一个节点
N_HIDDEN = 200  # 模拟网络复杂
N_SAMPLES = 20  # 模拟数据量小
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), dim=1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
x, y = Variable(x), Variable(y)
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), dim=1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
# volatile如果只有前向计算，没有后向梯度计算，设置volatile可以提高效率。
test_x, test_y = Variable(test_x, volatile=True), Variable(test_y, volatile=True)

# plt.scatter(test_x.data,test_y.data,c='red',s=50,alpha=0.5,label='test')
# plt.scatter(x.data,y.data,c='blue',s=50,alpha=0.5,label='train')
# plt.legend(loc='best')
# plt.show()

net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),#可以加载网络层和激励函数之间，也可以加载激励函数和网络层之间
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

optimizer_overfitting = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_dropped = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

for i in range(500):
    overfitting_predict = net_overfitting(x)
    dropped_predict = net_dropped(x)

    loss_over = loss_func(overfitting_predict, y)
    loss_drop = loss_func(dropped_predict, y)

    optimizer_overfitting.zero_grad()
    optimizer_dropped.zero_grad()

    loss_over.backward()
    loss_drop.backward()

    optimizer_overfitting.step()
    optimizer_dropped.step()

    if i % 100 == 0:
        net_overfitting.eval()
        net_dropped.eval()
        over_predict = net_overfitting(test_x)
        drop_predict = net_dropped(test_x)

        plt.scatter(test_x.data, test_y.data, c='red', s=50, alpha=0.5, label='test')
        plt.scatter(x.data, y.data, c='blue', s=50, alpha=0.5, label='train')

        plt.plot(test_x.data.numpy(), over_predict.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), drop_predict.data.numpy(), 'b--', lw=3, label='dropped')
        plt.text(0, -1.2, 'overfitting loss:%.4f' % loss_func(over_predict, test_y).data[0],
                 fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropped loss:%.4f' % loss_func(dropped_predict, test_y).data[0],
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='best')
        plt.show()
        net_dropped.train()
        net_overfitting.train()
