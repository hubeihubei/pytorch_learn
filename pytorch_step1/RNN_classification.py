import torch
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable

BATCH_SIZE = 50
DOWNLOAD_MNIST = True
TIME_STEP = 28
INPUT_SIZE = 28
EPOCH = 1
LR = 0.01
# train_data torch.Size([60000, 28, 28])
train_data = torchvision.datasets.MNIST(
    download=True,
    root='./mnist',
    transform=torchvision.transforms.ToTensor(),
    train=True
)
print(train_data.train_data.size())
data_loader = Data.DataLoader(dataset=train_data, shuffle=True, batch_size=BATCH_SIZE)
# test_data torch.Size([10000, 28, 28])
test_data = torchvision.datasets.MNIST(root='./mnist', train=False,transform=torchvision.transforms.ToTensor())
print(test_data.test_data.size())
# test_x torch.Size([2000, 1, 28, 28])
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000] / 255
test_x = Variable(test_data.test_data).type(torch.FloatTensor)[:2000] / 255
print(test_x.size())
test_y = test_data.test_labels[:2000]


class RNN(torch.nn.Module):
    # 入参：(batch_size,time_step,input_size)
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,
            # 如果让batch_size 作为第一个参数则True
            batch_first=True
        )
        self.out = torch.nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)
        # 取最后时刻的r_out  torch.Size([50, 64])

        y = self.out(r_out[:, -1, :])
        return y


rnn = RNN()


optimzer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(data_loader):
        b_x = Variable(b_x.view(-1, 28, 28))
        b_y = Variable(b_y)

        output = rnn(b_x)
        # print(output)
        loss = loss_func(output, b_y)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if step % 50 == 0:
            test_x = test_x.view(-1, 28, 28)
            test_output = rnn(test_x)
            prediction = torch.max(test_output, 1)[1].data
            accuracy = sum(prediction == test_y) / test_y.size(0)
            print("epoch：", epoch, "loss:", loss.data[0], "accuracy:", accuracy)

test_input = test_x[:10].view(-1, 28, 28)
test_output = rnn(test_input)
pred = torch.max(test_output, 1)[1].data.squeeze()
print("pred:", pred.view(1, -1))
print("really:", test_y[:10].view(1, -1))
