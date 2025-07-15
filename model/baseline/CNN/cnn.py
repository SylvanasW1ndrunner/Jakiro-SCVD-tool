import re

import torch
import torch.nn as nn


class CNN_Model(nn.Module):
    def __init__(self, input_dim):
        super(CNN_Model, self).__init__()
        # 直接使用全连接层（Linear）
        self.fc1 = nn.Linear(input_dim, 128)  # 将输入从 768 映射到 128
        self.fc2 = nn.Linear(128, 64)  # 再映射到 64
        self.fc3 = nn.Linear(64, 1)  # 输出层，1个节点，进行二分类
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数用于二分类任务

    def forward(self, x):
        # 通过全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)  # 输出 [0, 1] 的概率值
        return x
