import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
class GCN_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, nclass):
        super(GCN_Module, self).__init__()
        # Graph Convolutional Layer
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)

        self.conv2 = pyg_nn.GCNConv(hidden_dim, nclass)

    def forward(self, x, edge_index):
        """
        :param x: Node feature matrix [num_nodes, input_dim]
        :param edge_index: Edge indices in COO format [2, num_edges]
        """
        # Process graph structure with GCN to extract features
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.sigmoid(x)
        return x
