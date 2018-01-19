import argparse

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision.datasets import CIFAR10
# 所有预先训练的模型都希望以相同的方式归一化输入图像，也就是小批量的形状为（3 x H x W）的3通道RGB图像，其中H和W预计至少为224。
# 图像必须加载到[0，1]的范围内，然后用均值= [0.485,0.456,0.406]和std = [0.229,0.224,0.225]

params = argparse.ArgumentParser()
params.add_argument('--epoch', default=20, type=int)
params.add_argument('--bs', default=32, type=int)
params.add_argument('--num_workers', default=8, type=int)
params.add_argument('--lr', default=0.001, type=int)
opt = params.parse_args()
DOWNLOAD_DATASET = True
train_transform = torchvision.transforms.Compose([transforms.Scale(256), transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(224), transforms.ToTensor(),
                                                  transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
train_dataset = CIFAR10(root='../cifar10', train=True, transform=train_transform, download=DOWNLOAD_DATASET)
test_dataset = CIFAR10(root='../cifar10', train=False, transform=transforms.ToTensor, download=DOWNLOAD_DATASET)

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=opt.bs, shuffle=True,num_workers=opt.num_workers)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=opt.bs, shuffle=True,num_workers=opt.num_workers)
n_classes=len(train_dataset.dataset.size(0))
resNet = torchvision.models.resnet18(pretrained=True).cuda()
dim_in = resNet.fc.in_features
resNet.fc=nn.Linear(dim_in,n_classes)
optimization = torch.optim.Adam(resNet.parameters(), lr=opt.lr)
loss_func = torch.nn.CrossEntropyLoss().cuda()

for epoch in range(opt.epoch):
    print(epoch)
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()
        out = resNet(b_x)
        optimize = torch.max(out, 1)[1].cuda()
        loss = loss_func(out, b_y)
        optimization.zero_grad()
        loss.backward()
        optimization.step()
        acc = sum(optimize.data == b_y.data) / b_x.size(0)
        if step % 50 == 0:
            print('loss:', loss, "acc:", acc)


torch.save(resNet,"../model/resNet.pth")