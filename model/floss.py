import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification
        :param alpha: 正类权重 (float)
        :param gamma: 调制因子 (float)
        :param reduction: 损失的返回方式 ('none', 'mean', 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        :param predictions: 模型预测值 (batch_size, 1)，为概率值
        :param targets: 实际标签 (batch_size, 1)，值为 0 或 1
        """
        # 防止数值溢出
        predictions = torch.clamp(predictions, 1e-6, 1.0 - 1e-6)

        # 正负样本的损失
        pos_loss = -self.alpha * (1 - predictions) ** self.gamma * targets * torch.log(predictions)
        neg_loss = -(1 - self.alpha) * predictions ** self.gamma * (1 - targets) * torch.log(1 - predictions)

        # 总损失
        loss = pos_loss + neg_loss

        # 根据 reduction 返回
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss