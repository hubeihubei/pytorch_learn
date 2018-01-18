import time
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import argparse
from cat_vs_dog.dataset import h5DataSet
from cat_vs_dog.Net import classifier

params = argparse.ArgumentParser()
params.add_argument("--model", help='vgg,resnet152,inceptionv3', nargs="+", default=['vgg', 'resnet152', 'inceptionv3'])
params.add_argument('--epoch', default=20, type=int)
params.add_argument('--bs', default=32, type=int)
params.add_argument('--num_workers', default=8, type=int)
params.add_argument('--n_classes', default=2, type=int)
opt = params.parse_args()

root = '/home/wangyang/IdeaProjects/pytorch_learn/cat_and_dog/'
train_list_hd5f = [root + 'train_feature_{}.hd5f'.format(i) for i in opt.model]
print(train_list_hd5f)
val_list_hd5f = [root + 'val_feature_{}.hd5f'.format(i) for i in opt.model]
print(val_list_hd5f)
feature_dataset = {
    'train': h5DataSet(train_list_hd5f),
    'val': h5DataSet(val_list_hd5f)
}
feature_dataloader = {
    'train': Data.DataLoader(
        dataset=feature_dataset['train'], batch_size=opt.bs, shuffle=True, num_workers=opt.num_workers
    ),
    'val': Data.DataLoader(
        dataset=feature_dataset['val'], batch_size=opt.bs, shuffle=True, num_workers=opt.num_workers
    )
}

feature_dataset_size = {
    'train': feature_dataset['train'].dataset.size(0),
    'val': feature_dataset['val'].dataset.size(0),
}

use_gpu = torch.cuda.is_available()
dimension = feature_dataset['train'].dataset.size(1)
mynet = classifier(dimension, opt.n_classes)
loss_func = nn.CrossEntropyLoss()
if use_gpu:
    mynet = mynet.cuda()
    loss_func = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(mynet.parameters(), lr=1e-3)

running_loss = 0.0
running_accu = 0.0
since = time.time()
mynet.train()
for epoch in range(opt.epoch):
    print(epoch)
    for step, (b_x, b_y) in enumerate(feature_dataloader['train']):
        b_x = Variable(b_x)
        b_y = Variable(b_y)
        if use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        out = mynet(b_x)
        loss = loss_func(out, b_y)

        if use_gpu:
            prediction = torch.max(out, 1)[1].cuda()
        else:
            prediction = torch.max(out, 1)[1]
        running_loss = b_y.data.size(0) * loss.data[0]
        running_accu += sum(prediction.cpu().data == b_y.cpu().data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 and step != 0:
            print("loss:", running_loss / (opt.bs * step), "accu:", running_accu / (opt.bs * step))
eplise_time = time.time() - since
print("loss:", running_loss / feature_dataset_size['train'], 'accu:', running_accu / feature_dataset_size['train'],
      'Time:', eplise_time)

val_accu = 0.0
val_loss = 0.0
mynet.eval()
for step, (t_x, t_y) in enumerate(feature_dataloader['val']):
    t_x = Variable(t_x)
    t_y = Variable(t_y)
    if use_gpu:
        t_x = t_x.cuda()
        t_y = t_y.cuda()
    out = mynet(t_x)
    val_loss = loss_func(out, t_y)
    if use_gpu:
        val_prediction = torch.max(out, 1)[1].cuda()
    else:
        val_prediction = torch.max(out, 1)[1]
    val_loss = t_y.data.size(0) * val_loss.data[0]
    val_accu += sum(val_prediction.cpu().data == t_y.cpu().data)
print("loss:", val_loss / feature_dataset_size['val'], 'accu:', val_accu / feature_dataset_size['val'])
print("Finish Training")

torch.save(mynet, '../model/feature_model.pth')
