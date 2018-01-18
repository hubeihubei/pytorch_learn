import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import os
# vgg = models.vgg19()
# print(list(vgg.children()))
# inceptionv3 = models.resnet152()
# l = list(inceptionv3.children())
# feature = nn.Sequential(*list(inceptionv3.children()))
# print(feature)

# 特征提取预训练网络
class feature_net(nn.Module):
    def __init__(self, model):
        super(feature_net, self).__init__()

        if model == 'vgg':
            vgg = models.vgg19(pretrained=True)
            # 舍弃最后一层（全连接层）
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            # 在最后加一层平均池化层，将结果转为特征向量
            self.feature.add_module("global average", nn.AvgPool2d(9))
        elif model == 'inceptionv3':
            inceptionv3 = models.inception_v3(pretrained=True)
            # 可变参数入参，如果已经有list或tuple则在前面加*
            self.feature = nn.Sequential(*list(inceptionv3.children())[:-1])
            # inception_v3的第13层也是全连接，需要去除
            self.feature._modules.pop("13")
            self.feature.add_module("global average", nn.AvgPool2d(35))
        elif model == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            # resnet 倒数第二层为 AvgPool2d(),所以不需要再加
            self.feature = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        return x


# 全连接分类层
class classifier(nn.Module):
    def __init__(self,dim_in,n_classes):
        super(classifier, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(dim_in,1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000,n_classes)
        )

    def forward(self, x):
        x=self.layer1(x)
        return x


