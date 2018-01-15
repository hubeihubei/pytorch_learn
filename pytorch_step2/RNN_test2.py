import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

# 读取csv文件，并只使用第二列数据
# data_csv=pd.read_csv('../CSV/data.csv',usecols=[1])# shape (145, 1)
data_csv = pd.read_csv('../CSV/test01.csv', usecols=[1])
# data2_csv=data2_csv.dropna()
# data2_csv=data2_csv.values
# plt.plot(data2_csv)
# plt.show()

# 去除数据集中的na
data_csv = data_csv.dropna()
dataset = data_csv.values  # shape (144, 1)
# 转换为float32
dataset = dataset.astype('float32')
# 归一
max_value = np.max(dataset)
dataset = list(map(lambda x: x / max_value, dataset))  # list len:144


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i + look_back])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


dataX, dataY = create_dataset(dataset, 2)
# dataX.shape (142, 2, 1)
# dataY.shape (142, 1)

train_size = int(len(dataX) * 0.7)
train_x = dataX[:train_size]
train_y = dataY[:train_size]

dataSet = Data.TensorDataset(data_tensor=torch.from_numpy(train_x), target_tensor=torch.from_numpy(train_y))
dataloader = Data.DataLoader(dataset=dataSet, batch_size=64, shuffle=True)
test_x = dataX[train_size:]
test_y = dataY[train_size:]


class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size=2, hidden_size=6, num_layers=2,dropout=0.2)
        self.layer2 = nn.Linear(6, 1)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(-1, h)
        x = self.layer2(x)
        x.view(s, b, -1)
        return x


lstm = Lstm().cuda()
optimizer = torch.optim.Adam(lstm.parameters(), 0.001)
loss_func = nn.MSELoss().cuda()
loss_list = []
for epoch in range(100):
    print(epoch)
    a = 0
    for step, (b_x, b_y) in enumerate(dataloader):
        # print('b_x:',b_x)
        torch_x = b_x.view(-1, 1, 2)
        torch_y = b_y.view(-1, 1, 1)
        # print('torch_x',torch_x)
        datax = Variable(torch_x).cuda()
        datay = Variable(torch_y).cuda()
        pre = lstm(datax)
        loss = loss_func(pre, datay)
        optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.data[0])
        optimizer.step()
        if (step % 10 == 0):
            print(step)
            print('loss:', loss)

plt.figure()
plt.plot(loss_list)
plt.xlabel("step")
plt.ylabel("loss")
plt.figure()
lstm.eval()
torch_testx = torch.from_numpy(test_x.reshape(-1, 1, 2))
var_testx = Variable(torch_testx).cuda()
prediction = lstm(var_testx)

plt.plot(test_y.reshape(-1, 1), color='red', label='real')
plt.plot(prediction.cpu().data, color='blue', label='pre')

plt.legend(loc='best')
plt.show()
