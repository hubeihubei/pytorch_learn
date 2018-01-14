import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

# title:批数据训练
# BATCH_SIZE 可以不被训练集整除
BATCH_SIZE = 50
x = torch.linspace(-1, 1, 100)
x = torch.unsqueeze(x, dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
# 定义数据集 data_tensor训练集，target_tensor验证集
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
# 定义dataloader
# data_loader output(batch_size,1)
data_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    # 定义是否打乱训练集
    shuffle=True,
    # 定义几个进程提取batch_x,batch_y
    num_workers=2,
)
# x, y = Variable(x), Variable(y)
net1 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()
# 整个dataset训练1000次
for epoch in range(1000):
    # step 当前dataset中第几个batch数据，这里一个dataset有100个，BATCH_SIZE=50，step可能的取值就是0,1
    for step, (batch_x, batch_y) in enumerate(data_loader):
        # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
        #       batch_x.numpy(), '| batch y: ', batch_y.numpy())
        print(batch_x.size())
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        prediction = net1(batch_x)
        loss = loss_func(prediction, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

predict = net1(Variable(x))
loss = loss_func(predict, Variable(y))
print(loss)
plt.plot(x.numpy(), predict.data.numpy(), color='red', lw=5)
plt.scatter(x.numpy(), y.numpy())
plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
plt.show()
