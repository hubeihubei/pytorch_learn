import argparse
import os
import sys
import h5py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from .Net import feature_net

parse = argparse.ArgumentParser()
parse.add_argument('--model', required=True, help='vgg,inceptionv3,resnet152')
parse.add_argument('--phase', required=True, help='train,val')
parse.add_argument('--bs', required=True, help='batch_size', default=32)
opt = parse.parse_args()


transform_compose = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

}
root = '/home/wangyang/IdeaProjects/pytorch_learn/cat_and_dog/'
data_folder = {
    "train": ImageFolder(root=root + "train/", transform=transform_compose['train']),
    'val': ImageFolder(root=root + "val/", transform=transform_compose['val'])
}

dataLoader = {
    'train': Data.DataLoader(
        dataset=data_folder['train'],
        batch_size=opt.bs,
        # 提取特征向量时，数据是不能随机打乱的，因为使用多个模型，每次随机大乱斗会造成标签混乱
        shuffle=False,
        num_workers=4
    ),
    'val': Data.DataLoader(
        dataset=data_folder['val'],
        batch_size=opt.bs,
        shuffle=False,
        num_workers=4
    )
}

use_gpu = torch.cuda.is_available()


def CreateFeature(model, phase, outpath='.'):
    if use_gpu:
        net = feature_net(model).cuda()
    else:
        net = feature_net(model)

    feature_map = torch.FloatTensor()
    label_map = torch.LongTensor()
    for step, (b_x, b_y) in enumerate(dataLoader['train']):
        if use_gpu:
            b_x = Variable(b_x, volatile=True).cuda()
        else:
            b_x = Variable(b_x, volatile=True)
        out = net(b_x)
        feature_map = torch.cat((feature_map, out.cpu().data), 0)
        label_map = torch.cat((label_map, b_y), 0)

    feature_map = feature_map.numpy()
    label_map = label_map.numpy()

    fileName = "_feature_{}.hd5f".format(model)
    h5py_path = os.path.join(outpath, phase) + fileName

    with h5py.File(h5py_path, 'w') as h:
        h.create_dataset('data', data=feature_map)
        h.create_dataset('label', data=label_map)


CreateFeature(opt.model, opt.phase)
