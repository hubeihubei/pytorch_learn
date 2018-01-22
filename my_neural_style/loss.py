import torch
import torch.nn as nn


class Content_loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        gram = torch.mm(features, features.t())
        # 标准化
        gram /= (a * b * c * d)
        return gram

class Style_loss(nn.Module):
    def __init__(self,target,weight):
        super(Style_loss, self).__init__()
        self.weight=weight
        self.target=target.detach()*self.weight
        self.gram=Gram()
        self.creterion=nn.MSELoss()

    def forward(self, input):
        G=self.gram(input)*self.weight
        self.loss=self.creterion(G,self.target)
        out=input.clone()
        return out

    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss