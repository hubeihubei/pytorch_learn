import torch
from torch.autograd import Variable
import torch.nn.functional as  func
import matplotlib.pyplot as plt
import torch.utils.data as Data

LR = 0.01
BATCH_SIZE = 50
EPOCH = 12
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# plt.scatter(x, y)
# plt.show()
dataset = Data.TensorDataset(target_tensor=y, data_tensor=x)
dataloader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.prediction = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = func.relu(self.hidden(x))
        y = self.prediction(x)
        return y


net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
opt_SGD = torch.optim.SGD(net_SGD.parameters(), LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), LR, momentum=0.9)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), LR, betas=(0.9, 0.999))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his=[[],[],[],[]]

for epoch in range(EPOCH):
    print(epoch)
    for step,(b_x,b_y) in enumerate(dataloader):
        b_x,b_y=Variable(b_x),Variable(b_y)
        for net,opt,l in zip(nets,optimizers,losses_his):
            prediction=net(b_x)
            loss=loss_func(prediction,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l.append(loss.data[0])

labels=['SGD', 'Momentum', 'RMSprop', 'Adam']
for i,l_his in enumerate(losses_his):
    plt.plot(l_his,label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0,0.2)
plt.show()
