import torch
from torch import nn
from torch.nn import functional as F


class ContentLoss(nn.Module):
    """
    内容损失，使用nn.Module,意味着这个类可以看作一个层
    """

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input: torch.Tensor):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):

    def gram_matrix(self, input: torch.Tensor):
        """
        计算gram矩阵,
        gram矩阵的计算公式为:
        G_{XL} = F_{XL}^T * F_{XL}
        """
        a, b, c, d = input.size()  # size结构为: (batch_size, channel, height, width)
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())  # 计算gram矩阵
        return G.div(a * b * c * d)

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input: torch.Tensor):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
