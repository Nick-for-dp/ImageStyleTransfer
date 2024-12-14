import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.criterion = nn.MSELoss()
        self.loss = torch.tensor(0.0, requires_grad=True)

    def forward(self, x):
        self.loss = self.criterion(x.clone(), self.target)
        return x

    def backward(self):
        # 保留计算图
        self.loss.backward(retain_graph=True)
        return self.loss


def gram(x):
    batch_size, c, h, w = x.size()  # c是卷积数
    f = x.reshape(batch_size * c, h * w)
    g = f @ f.T / (batch_size * c * h * w)
    return g


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.criterion = nn.MSELoss()
        self.loss = torch.tensor(0.0, requires_grad=True)

    def forward(self, x):
        target_gram = gram(self.target)
        x_gram = gram(x)
        self.loss = self.criterion(target_gram, x_gram)
        return x

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss
