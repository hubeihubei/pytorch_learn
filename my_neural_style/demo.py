import torch
from torch.autograd import Variable
from my_neural_style.load_img import load_img, show_img
from my_neural_style.run_code import run_style_transfer

content_img = load_img('./picture/content.png')
content_img = Variable(content_img).cuda()
style_img = load_img('./picture/style.png')
style_img = Variable(style_img).cuda()
input_img = content_img.clone()
input_param = run_style_transfer(content_img, style_img, input_img)
show_img(input_param.cpu())
