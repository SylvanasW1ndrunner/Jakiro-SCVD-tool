import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout

        # 第一层 GraphSAGE
        self.conv1 = SAGEConv(nfeat, nhid)
        # 第二层 GraphSAGE
        self.conv2 = SAGEConv(nhid, nclass)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

