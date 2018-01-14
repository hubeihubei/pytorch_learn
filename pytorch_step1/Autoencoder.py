import matplotlib
import torchvision
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

EPOCH = 15
LR = 0.005
BATCH_SIZE = 50
N_TEST_img = 5

train_data = torchvision.datasets.MNIST(
    train=True, transform=torchvision.transforms.ToTensor(), root='./mnist', download=False
)
data_loader = Data.DataLoader(
    batch_size=BATCH_SIZE, shuffle=True, dataset=train_data
)


# print(train_data.train_data[0])
# plt.imshow(train_data.train_data[0],cmap='gray')
# plt.show()

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 3),
            # torch.nn.Tanh(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded)
        decoded = self.decoder(encoded)
        return encoded, decoded


auto_encoder = AutoEncoder()
# print(auto_encoder)
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()
view_data = train_data.train_data[0:N_TEST_img]
for epoch in range(EPOCH):
    print(epoch)
    for step, (x, y) in enumerate(data_loader):
        # print(x.size())
        # print(type(x.view(-1, 28 * 28)))
        b_x = Variable(x.view(-1, 28 * 28))
        b_y = Variable(x.view(-1, 28 * 28))
        # print(b_x.size())
        encoded, decoded = auto_encoder(b_x)
        decoded.cuda()
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 and epoch in (0, 5, EPOCH - 1):
            # print(type(view_data.view(-1,28*28)))
            v_view_data = Variable(view_data.view(-1, 28 * 28).type(torch.FloatTensor) / 255)
            _, decoded_data = auto_encoder(v_view_data)
            # 2行，5列,图片大小长6，宽2
            f, a = plt.subplots(2, N_TEST_img, figsize=(5, 2))
            for i in range(N_TEST_img):
                a[0][i].imshow(view_data[i], cmap='gray')
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())

            for i in range(N_TEST_img):
                a[1][i].imshow(decoded_data.data[i].view(28, 28), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())

            plt.show()
fig = plt.figure(2)
v_view_data2 = Variable(train_data.train_data[0:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255)
encoded_data, _ = auto_encoder(v_view_data2)
# 构建3d图
ax = Axes3D(fig)
# print(encoded_data.data)
# x,y,z 坐标的集合
X, Y, Z = encoded_data.data[:, 0], encoded_data.data[:, 1], encoded_data.data[:, 2]
# 对应的label集合
labels = train_data.train_labels[:200]

for x, y, z, label in zip(X, Y, Z, labels):
    # 构建颜色对应的公式，0-255（label属于0-9 除以9得到0-1再乘以255）
    c = cm.rainbow(int(255 * label / 9))
    ax.text(x, y, z, label, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
plt.show()
print('end')
