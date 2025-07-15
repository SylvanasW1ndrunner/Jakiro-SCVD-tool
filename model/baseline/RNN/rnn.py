import torch
import torch.nn as nn

class RNN_Model(nn.Module):
    def __init__(self, input_dim):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNN(input_dim, 512, batch_first=True)
        self.fc = nn.Linear(512, 1)  # 1 output for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.rnn(x)
        x = self.fc(h_n[-1])
        x = self.sigmoid(x)  # Sigmoid activation for binary classification
        return x
