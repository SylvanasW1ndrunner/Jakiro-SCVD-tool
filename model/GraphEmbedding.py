import networkx as nx
import torch
import pygraphviz as pgv
from torch_geometric import nn


class GraphEncoder(nn.Module):
    def __init__(self, model):
        super(GraphEncoder, self).__init__()
        self.model = model
        self.graphs = []

        # 从给定的图数据中创建图对象
        for graph in self.graphdata:
            graph = pgv.AGraph(data=graph)
            G = nx.nx_agraph.from_agraph(graph)
            self.graphs.append(G)

    def foward(self):
        all_node_features = []
        all_adjacency_matrix = []

        # 遍历每个图进行处理
        for graph in self.graphs:
            nodes = graph.nodes(data=True)
            edges = graph.edges()
            node_features = {}
            edge_features = {}

            # 生成节点特征
            for node_id, data in nodes:
                feature = data['label']  # 假设节点特征存储在 'label' 字段
                node_features[node_id] = torch.tensor(self.model.get_sentence_vector(feature))

            # 生成邻接矩阵，边的存在用 1 表示，边的不存在用 0 表示
            num_nodes = len(graph.nodes())
            adj_matrix = torch.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵

            # 填充邻接矩阵
            for edge in edges:
                u, v = edge
                u_idx = list(graph.nodes()).index(u)  # 获取节点 u 的索引
                v_idx = list(graph.nodes()).index(v)  # 获取节点 v 的索引
                adj_matrix[u_idx, v_idx] = 1
                adj_matrix[v_idx, u_idx] = 1  # 对于无向图，邻接矩阵是对称的

            # 将每个图的节点特征和邻接矩阵加入到列表中
            all_node_features.append(node_features)
            all_adjacency_matrix.append(adj_matrix)

        return all_node_features, all_adjacency_matrix
