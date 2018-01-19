import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from PIL import Image

img_path = "/home/wangyang/IdeaProjects/cat_and_dog/val/cat/cat.157.jpg"
transform1=transforms.Compose([transforms.Scale(320),transforms.CenterCrop(299),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img=Image.open(img_path).convert('RGB')
img_tensor=transform1(img)
resNet=torch.load('/home/wangyang/IdeaProjects/pytorch_learn/model/resNet.pth')
out=resNet(img_tensor)
prediction=torch.max(out,1)[1].cuda()
print(prediction)