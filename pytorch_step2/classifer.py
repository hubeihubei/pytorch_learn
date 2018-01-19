import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torch.utils.data as Data

params = argparse.ArgumentParser()
params.add_argument('--img_path',default='/home/wangyang/IdeaProjects/cat_and_dog/val/cat/cat.157.jpg')
params.add_argument('--model_path',default='/home/wangyang/IdeaProjects/pytorch_learn/model/resNet.pth')
params.add_argument('--bs', default=32, type=int)
opt = params.parse_args()

def class_map(key):
    map = {0: 'alplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
           9: 'truck'}
    return map[key]


def classifer(img_path, model_path, batch_size):
    resNet = torch.load(model_path).cuda()
    transform1 = transforms.Compose([transforms.Scale(320), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_folder = ImageFolder(root=img_path, transform=transform1),
    dataloader = Data.DataLoader(
        dataset=img_folder,
        batch_size=batch_size,
        # 提取特征向量时，数据是不能随机打乱的，因为使用多个模型，每次随机大乱斗会造成标签混乱
        shuffle=False,
        num_workers=4
    )

    for step, (b_x, b_y) in enumerate(dataloader):
        v_tensor = Variable(b_x.cuda())
        out = resNet(v_tensor)
        prediction = torch.max(out, 1)[1].cuda()
        prediction.cpu().data.view(1, -1)
        class_list = list(prediction)
        result=list(map(class_map, class_list))
    return result


if __name__ == '__main__':
    img_path = opt.img_path
    model_path = opt.model_path
    batch_size = opt.bs
    classifer(img_path, model_path, batch_size)
