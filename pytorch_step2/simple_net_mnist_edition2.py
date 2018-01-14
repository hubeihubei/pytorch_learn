import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable

LR = 0.001
BATCH_SIZE = 64
EPOCH = 10
transform_compose = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5])
])
train_data = torchvision.datasets.MNIST(
    root="../mnist",
    train=True,
    transform=transform_compose,
    download=False
)
test_data = torchvision.datasets.MNIST(
    root="../mnist",
    train=False,
    transform=transform_compose,

)
train_dataloader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # b 16 28 28
            nn.BatchNorm1d(16),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),  # b 32 28 28
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # b 32 14 14
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # b 64 14 14
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # b 128 14 14
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # b 128 7 7
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


net = Net().cuda()
optimization = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print(epoch)
    for step, (b_x, b_y) in enumerate(train_dataloader):
        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()
        prediction = net(b_x)
        loss = loss_func(prediction, b_y)
        optimization.zero_grad()
        loss.backward()
        optimization.step()
        accury=torch.max(prediction,1)[1].cuda()
        acc = sum(b_y.data == accury.data) / b_x.data.size(0)
        if step% 1000 ==0:
            print("loss:", loss, "acc:", acc)

eval_loss = 0
eval_acc = 0
# for data in test_dataloader:
#     t_x,t_y=data
#     net.eval()
#     t_x=Variable(t_x,volatile=True).cuda()
#     t_y=Variable(t_y,volatile=True).cuda()
#     out=net(t_x)
#     loss_test=loss_func(out,t_y)
#     predict_test=torch.max(out,1)[1].cuda()
#     eval_loss+=loss_test.data[0]*t_x.size(0)
#     eval_acc+=sum(predict_test.data==t_y.data)
#
# print("Test loss:",eval_loss/len(test_data),"Test acc:",eval_acc/len(test_data))
net.eval()
test_data_sample = Variable(test_data.test_data.view(-1, 1, 28, 28)).float()[:10].cuda()
print("shape:", test_data_sample.size())
out_result = net(test_data_sample)
predict_result = torch.max(out_result, 1)[1].data
print("predict:", predict_result, "really:", test_data.test_labels[:10])
