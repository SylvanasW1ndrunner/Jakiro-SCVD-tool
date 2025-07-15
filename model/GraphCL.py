import torch
import random

from torch_geometric import nn
from torch_geometric.utils import dropout_node, dropout_edge, subgraph
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_networkx
import networkx as nx
import torch.nn.functional as F

class GraphAugmentations:
    def __init__(self, node_drop_prob=0.2, edge_perturb_prob=0.2, attr_mask_prob=0.2, subgraph_ratio=0.8):
        self.node_drop_prob = node_drop_prob
        self.edge_perturb_prob = edge_perturb_prob
        self.attr_mask_prob = attr_mask_prob
        self.subgraph_ratio = subgraph_ratio

    def node_dropping(self, data):
        # 使用 dropout_node 进行节点丢弃
        edge_index, node_mask = dropout_node(data.edge_index, p=self.node_drop_prob, num_nodes=data.num_nodes)
        data.x = data.x[node_mask]  # 更新节点特征
        data.edge_index = edge_index  # 更新边索引
        data.num_nodes = data.x.size(0)
        return data

    def edge_perturbation(self, data):
        # 使用 dropout_edge 进行边删除
        edge_index, _ = dropout_edge(data.edge_index, p=self.edge_perturb_prob)
        num_nodes = maybe_num_nodes(edge_index, num_nodes=data.num_nodes)

        # 随机添加一定比例的边
        num_edges_to_add = int(data.edge_index.size(1) * self.edge_perturb_prob)
        possible_edges = torch.combinations(torch.arange(num_nodes), r=2).t()
        existing_edges = set(map(tuple, data.edge_index.t().tolist()))
        possible_edges_set = set(map(tuple, possible_edges.t().tolist()))
        new_edges = list(possible_edges_set - existing_edges)
        if len(new_edges) > 0:
            new_edges = random.sample(new_edges, min(num_edges_to_add, len(new_edges)))
            new_edges = torch.tensor(new_edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, new_edges], dim=1)

        data.edge_index = edge_index
        return data

    def feature_masking(self, data):
        # 随机遮蔽部分节点的特征
        num_nodes = data.x.size(0)
        mask = torch.rand(num_nodes) < self.attr_mask_prob
        data.x[mask] = 0
        return data

    def subgraph_sampling(self, data):
        # 通过随机游走采样获取子图
        num_nodes = data.num_nodes
        sub_num_nodes = int(num_nodes * self.subgraph_ratio)
        if sub_num_nodes < 2:
            return data  # 子图节点数过少，返回原图

        # 转换为 networkx 图进行随机游走采样
        G = to_networkx(data, to_undirected=True)
        start_node = random.choice(list(G.nodes))
        lengths = dict(nx.single_source_shortest_path_length(G, start_node, cutoff=sub_num_nodes))
        sub_nodes = list(lengths.keys())
        sub_edge_index, _ = subgraph(sub_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

        data.x = data.x[sub_nodes]
        data.edge_index = sub_edge_index
        data.num_nodes = data.x.size(0)
        return data

    def apply_transforms(self, data):
        # 随机选择一种增强方式进行应用
        augmentation_methods = [self.node_dropping, self.edge_perturbation, self.feature_masking, self.subgraph_sampling]
        method = random.choice(augmentation_methods)
        data = method(data)
        return data


# 图卷积编码器 (GCN)
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNEncoder, self).__init__()
        self.gcn1 = nn.Linear(input_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        # 第1层图卷积
        h = F.relu(self.gcn1(torch.matmul(adj, x)))  # 使用邻接矩阵乘以特征
        # 第2层图卷积
        h = F.relu(self.gcn2(torch.matmul(adj, h)))
        return h

# 图注意力编码器 (GAT)
class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super(GATEncoder, self).__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        # 将邻接矩阵与特征矩阵结合进行注意力计算
        attn_output_1, _ = self.attn1(x, x, x)
        h = F.relu(self.fc1(attn_output_1))

        attn_output_2, _ = self.attn2(h, h, h)
        h = F.relu(self.fc2(attn_output_2))
        return h

# 图内部对比学习模块
class GraphContrastiveLearning(nn.Module):
    def __init__(self, gcn_encoder, gat_encoder, hidden_dim):
        super(GraphContrastiveLearning, self).__init__()
        self.gcn_encoder = gcn_encoder
        self.gat_encoder = gat_encoder
        self.projector = nn.Linear(hidden_dim, hidden_dim)  # 投影层

    def forward(self, x1, adj1, x2, adj2):
        # 分别对两个增强视图使用GNN和GAT编码器编码
        gcn_output_1 = self.gcn_encoder(x1, adj1)
        gat_output_2 = self.gat_encoder(x2, adj2)

        # 投影输出，用于对比学习
        z1 = self.projector(gcn_output_1)
        z2 = self.projector(gat_output_2)

        return z1, z2

    def contrastive_loss(self, z1, z2, tau=0.5):
        # InfoNCE损失
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.mm(z1, z2.t()) / tau
        labels = torch.arange(z1.size(0)).long().to(z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss

# 图模态对比学习模块整合
class GraphCLWithMultipleEncoders(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphCLWithMultipleEncoders, self).__init__()
        self.augmentor = GraphAugmentations()  # 图增强模块
        self.gcn_encoder = GCNEncoder(input_dim, hidden_dim)
        self.gat_encoder = GATEncoder(input_dim, hidden_dim)
        self.contrastive_module = GraphContrastiveLearning(self.gcn_encoder, self.gat_encoder, hidden_dim)

    def forward(self, graph):
        # 生成两个增强视图
        graph1 = self.augmentor.apply_transforms(graph)
        graph2 = self.augmentor.apply_transforms(graph)

        # 获取增强视图的特征矩阵和邻接矩阵
        x1, adj1 = graph1.features, graph1.adj
        x2, adj2 = graph2.features, graph2.adj

        # 对比学习
        z1, z2 = self.contrastive_module(x1, adj1, x2, adj2)
        return z1, z2

    def graph_cl_loss(self, z1, z2):
        return self.contrastive_module.contrastive_loss(z1, z2)