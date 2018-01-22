import torch
import torch.nn as nn
import torchvision.models as models
from my_neural_style import loss

# 只保留前面的卷积层，后面的全连接层去掉
vgg = models.vgg19(pretrained=True).features
if torch.cuda.is_available():
    vgg = vgg.cuda()

content_layer_default = ['conv4']
style_layer_default = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']


def get_style_model_and_loss(style_image, content_image, content_weight=1, style_weight=1000, cnn=vgg,
                             content_layers=content_layer_default, style_layers=style_layer_default):
    content_loss_list = []
    style_loss_list = []
    gram = loss.Gram()
    if torch.cuda.is_available():
        gram = gram.cuda()
    model = nn.Sequential()
    if torch.cuda.is_available():
        model = model.cuda()
    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv' + str(i)
            model.add_module(name, layer)
            if name in content_layers:
                target = model(content_image)
                content_loss = loss.Content_loss(target, content_weight)
                model.add_module('content_loss' + str(i), content_loss)
                content_loss_list.append(content_loss)

            if name in style_layers:
                target = model(style_image)
                target = gram(target)
                style_loss = loss.Style_loss(target, style_weight)
                model.add_module('style_loss' + str(i), style_loss)
                style_loss_list.append(style_loss)
            i += 1
        if isinstance(layer, nn.ReLU):
            model.add_module('relu' + str(i), layer)
        if isinstance(layer, nn.MaxPool2d):
            model.add_module('pool' + str(i), layer)
    return model,style_loss_list,content_loss_list