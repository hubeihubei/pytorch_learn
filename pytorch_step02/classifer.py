import argparse

import time
import torch
import h5py
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import torch.utils.data as Data
# import sys
# sys.path.append('.')
from pytorch_step02.my_dataset import MyDataSet
import os

params = argparse.ArgumentParser()
params.add_argument('--img_path', default='/home/wangyang/IdeaProjects/test_data')
params.add_argument('--model_path', default='/home/wangyang/IdeaProjects/pytorch_learn/model/resNet.pth')
params.add_argument('--bs', default=10, type=int)
opt = params.parse_args()

h5py_path_root = '/home/wangyang/IdeaProjects/pytorch_learn/h5py_data'


def class_map(key):
    map = {0: 'alplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
           9: 'truck'}
    return map[key]


def classifer(img_dir_path, model_path, batch_size):
    resNet = torch.load(model_path).cuda()
    transform_compose = transforms.Compose([transforms.Scale(320), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_list = os.listdir(img_dir_path)
    temp_list = torch.FloatTensor()
    img_names = []
    for image in image_list:
        image_path = os.path.join(img_dir_path, image)
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform_compose(img)
        tensor=img_tensor.view(1,-1,224)
        temp_list = torch.cat((temp_list, tensor), 0)
        img_names.append(image)
    h5py_name = "_h5py_name_{}.hd5f".format(str(time.time())[:10])
    h5py_path = os.path.join(h5py_path_root, h5py_name)
    if(not os.path.exists(h5py_path_root)):
        os.mkdir(h5py_path_root)
    with h5py.File(h5py_path, 'w') as h:
        h.create_dataset('data', data=temp_list.numpy())
        h.create_dataset('label', data=np.zeros((temp_list.size(0),1)))

    dataset = MyDataSet(h5py_path)
    dataloader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    result=[]

    for step, (b_x, _) in enumerate(dataloader):
        b_x=b_x.view(b_x.size(0),3,-1,224)
        v_tensor = Variable(b_x.cuda())
        out = resNet(v_tensor)
        prediction = torch.max(out, 1)[1].cuda()
        prediction=prediction.cpu().data.view(1,-1)
        class_list = prediction.numpy().tolist()[0]
        result += list(map(class_map, class_list))

    result_map=dict(zip(img_names,result))
    return result_map


if __name__ == '__main__':
    img_path = opt.img_path
    model_path = opt.model_path
    batch_size = opt.bs
    result = classifer(img_path, model_path, batch_size)
    print(result)
