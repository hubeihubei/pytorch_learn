import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.nn import init as init
import numpy as np

BATCH_SIZE = 64
N_SAMPLES = 2000
ACTIVATION = func.tanh
LR = 0.03
B_INIT = -0.2
N_HIDDEN = 8
EPOCH = 6

x = torch.linspace(-7, 10, N_SAMPLES).view(-1, 1)
noise = np.random.normal(0, 2, x.size())
y = torch.pow(x, 2).view(-1, 1) + torch.from_numpy(noise).float() - 5
# plt.scatter(x,y)
# plt.show()
dataset = Data.TensorDataset(data_tensor=x.float(), target_tensor=y.float())
dataloader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_x = torch.linspace(-7, 10, N_SAMPLES).view(-1, 1)
noise = np.random.normal(0, 2, test_x.size())
test_y = torch.pow(x, 2).view(-1, 1) + torch.from_numpy(noise).float() - 5
test_x, test_y = Variable(test_x).float(), Variable(test_y).float()


class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)

        for i in range(N_HIDDEN):
            if i == 0:
                input_size = 1
            else:
                input_size = 10
            fc = nn.Linear(input_size, 10)
            self._set_init(fc)
            self.fcs.append(fc)
            setattr(self, 'fc%i' % i, fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)

        self.prediction = nn.Linear(10, 1)
        self._set_init(self.prediction)

    def _set_init(self, layer):
        # 将y=wx+β 的w（权重）按照正态分布微调，β被替换为B_INIT
        init.normal(layer.weight, mean=0., std=.1)
        # 将layer.bias 这个Tensor的val 全部替换为B_INIT
        init.constant(layer.bias, B_INIT)

    def forward(self, x):
        if self.do_bn:
            x = self.bn_input(x)
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            if self.do_bn:
                x = self.bns[i](x)
            x = ACTIVATION(x)

        out = self.prediction(x)
        return out


nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
optimizers = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]
loss_func = torch.nn.MSELoss()
loss_list = [[], []]
for epoch in range(EPOCH):
    print(epoch)
    for step, (batch_x, batch_y) in enumerate(dataloader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        for net, optimizer in zip(nets, optimizers):
            out = net(b_x)
            loss = loss_func(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for net, l in zip(nets, loss_list):
        net.eval()
        pred = net(test_x)
        l.append(loss_func(pred, test_y).data[0])
        net.train()

plt.figure(1)
plt.plot(loss_list[0], c='red', lw=1, label='fn')
plt.plot(loss_list[1], c='blue', lw=1, label='bn')
plt.xlabel('step')
plt.ylabel('loss')
plt.ylim((0, 2000))
plt.legend(loc='best')
plt.figure(2)
nets[1].eval()
nets[0].eval()
bn_predict = nets[1](test_x)
fn_predict = nets[0](test_x)
plt.scatter(test_x.data, test_y.data, c='red', alpha=0.5, s=50, label='test')
plt.plot(test_x.data.numpy(), fn_predict.data.numpy(), 'r-', lw=3, label='fn')
plt.plot(test_x.data.numpy(), bn_predict.data.numpy(), 'b--', lw=3, label='bn')
plt.legend(loc='best')
plt.show()
