import shutil

import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import torchvision.models as models
import os
import torchvision.transforms as transforms


# 将训练集中的猫和狗图片进行分类
def image_classifier():
    root = "/home/wangyang/IdeaProjects/pytorch_learn/cat_and_dog/zip"
    val_path = "/home/wangyang/IdeaProjects/pytorch_learn/cat_and_dog/val"
    train_path = "/home/wangyang/IdeaProjects/pytorch_learn/cat_and_dog/train"
    datafiles = os.listdir(root)
    dog_files = list(filter(lambda x: x[:3] == 'dog', datafiles))
    cat_files = list(filter(lambda x: x[:3] == 'cat', datafiles))

    for i in range(len(dog_files)):
        pic_path = root + '/' + dog_files[i]
        if i < len(dog_files) * 0.9:
            obj_path = train_path + "/dog/" + dog_files[i]
        else:
            obj_path = val_path + "/dog/" + dog_files[i]
        shutil.move(pic_path, obj_path)

    for i in range(len(cat_files)):
        pic_path = root + '/' + cat_files[i]
        if i < len(cat_files) * 0.9:
            obj_path = train_path + "/cat/" + cat_files[i]
        else:
            obj_path = val_path + "/cat/" + cat_files[i]
        shutil.move(pic_path, obj_path)


# image_classifier()
def train():
    BATCH_SIZE = 32
    EPOCH = 5
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Scale(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_path = "/home/wangyang/IdeaProjects/pytorch_learn/cat_and_dog/train/"
    data_folder = {
        'train': torchvision.datasets.ImageFolder(root='../cat_and_dog/train/', transform=data_transforms['train']),
        'val': torchvision.datasets.ImageFolder(root='../cat_and_dog/val/', transform=data_transforms['val'])
    }

    data_loader = Data.DataLoader(dataset=data_folder['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # print(len(data_loader.dataset.imgs))
    val_loader = Data.DataLoader(dataset=data_folder['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # 获得训练集中分了几类
    img_classes = len(data_loader.dataset.classes)
    # pretrained (bool) – True, 返回在ImageNet上训练好的模型
    model = models.resnet18(pretrained=True)
    # 是否固定参数
    fix_param = False

    use_gpu = torch.cuda.is_available()

    if fix_param:
        for param in model.parameters():
            # 在训练时如果想要固定网络的底层，那么可以令这部分网络对应子图的参数requires_grad为False
            param.requires_grad = False

    dim_in = model.fc.in_features  # 最后fc层的输入
    model.fc = nn.Linear(dim_in, img_classes)
    if use_gpu:
        model = model.cuda()
        loss_func = nn.CrossEntropyLoss().cuda()
    else:
        loss_func = nn.CrossEntropyLoss()
    if fix_param:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    since = time.time()
    for epoch in range(EPOCH):
        print(epoch)
        running_loss = 0.0
        running_accu = 0.0
        model.train()
        for step, (b_x, b_y) in enumerate(data_loader):
            # print(b_y)
            if use_gpu:
                b_x = Variable(b_x).cuda()
                b_y = Variable(b_y).cuda()
            else:
                b_x = Variable(b_x)
                b_y = Variable(b_y)
            out = model(b_x)
            loss = loss_func(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_gpu:
                prediction = torch.max(out, 1)[1].cuda()
            else:
                prediction = torch.max(out, 1)[1]
            running_loss += loss.data[0] * b_y.data.size(0)
            running_accu += sum(b_y.cpu().data == prediction.cpu().data)
            if step % 25 == 0 and step != 0:
                print("loss:", running_loss / (b_y.data.size(0) * step), "accu:", running_accu / (b_y.data.size(0) * step))
    elips_time = time.time() - since
    running_loss /= len(data_loader.dataset)
    running_accu /= len(data_loader.dataset)

    print('Loss: {:.6f}, Acc: {:.4f}, Time: {:.0f}s'.format(
        running_loss, running_accu, elips_time))
    torch.save(model,'../model/cat_vs_dog.pkl')
    print('Validation')
    model.eval()
    eval_loss = 0.0
    eval_accu = 0.0
    for step, (t_x, t_y) in enumerate(val_loader):
        if use_gpu:
            t_x = Variable(t_x).cuda()
            t_y = Variable(t_y).cuda()
        else:
            t_x = Variable(t_x)
            t_y = Variable(t_y)
        val_out = model(t_x)
        val_loss = loss_func(val_out, t_y)
        if use_gpu:
            pre = torch.max(val_out, 1)[1].cuda()
        else:
            pre = torch.max(val_out, 1)[1]
        eval_loss += val_loss.data[0] * BATCH_SIZE
        eval_accu += sum(t_y.cpu().data == pre.cpu().data)

    print("eval loss:",eval_loss/len(val_loader.dataset),'eval accu:',eval_accu/len(val_loader.dataset))




# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomSizedCrop(299),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]),
#     'val': transforms.Compose([
#         transforms.Scale(320),
#         transforms.CenterCrop(299),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
# }
#
# model=torch.load("../model/cat_vs_dog.pkl")
# test_dataset=torchvision.datasets.ImageFolder(root='../cat_and_dog/val/', transform=data_transforms['val'])
# test_dataloader=Data.DataLoader(dataset=test_dataset,batch_size=10,shuffle=True)
# pre_list=[]
# rel_list=[]
# for i,(b_x,b_y)in enumerate(test_dataloader):
#     out=model(Variable(b_x).cuda())
#     pre=torch.max(out,1)[1].cuda()
#     pre_list.append(pre)
#     rel_list.append(b_y)
#     if i ==0:
#         break
# print("rel_list:",rel_list,"pre_list:",pre_list)

if __name__ == '__main__':
    train()