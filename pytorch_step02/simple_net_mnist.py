import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.utils.data as Data

LR = 0.01
BATCH_SIZE = 64
EPOCH =12
# transforms.Compose()将各种预处理操作组合到一起，transforms.ToTensor()，就是将图片转换成Tensor，
# 在转化过程中pytorch自动将图片标准化，也就是说Tensor范围是0~1
# transforms.Normalize([],[]),需要传两个参数：第一个参数是均值，第二个参数是方差，做的处理就是减均值，再除以方差，
# 因为是灰度图，所以只有一个通道，如果是彩色图有三个通道，transforms.Normalize([a,b,c],[d,e,f])
data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='../mnist', train=True, transform=data_tf, download=False)
test_dataset = datasets.MNIST(root='../mnist', train=False, transform=data_tf)

train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class simpleNet(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden1, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        return out


net = simpleNet(28 * 28, 300, 100, 10).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print(epoch)
    loss_n = 0
    for step, (b_x, b_y) in enumerate(train_dataloader):
        b_x = b_x.view(b_x.size(0), -1)
        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()
        out = net(b_x)
        loss = loss_func(out, b_y)
        loss_n = loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('epoch:', epoch, 'loss:', loss_n)

net.eval()
eval_acc = 0
eval_loss = 0
for data in test_dataloader:
    t_x, t_y = data
    t_x = Variable(t_x.view(t_x.size(0), -1), volatile=True).cuda()
    t_y = Variable(t_y, volatile=True).cuda()
    out = net(t_x)
    loss = loss_func(out, t_y)
    prediction = torch.max(out, 1)[1].cuda()
    acc = sum(prediction.data == t_y.data)
    eval_acc += acc
    eval_loss +=loss.data[0]*t_x.size(0)
# print(type(eval_acc),type(eval_loss))
# print(len(eval_acc))
print('Test Loss:',eval_loss/len(test_dataset),'Test acc:',eval_acc/len(test_dataset))
