import pickle
from sched import scheduler

import numpy as np
import torch_geometric
from fontTools.misc.timeTools import epoch_diff
from pyflakes.checker import counter
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from SmartContractDetect.CFGGen import CFGGen, read_dot
from SmartContractDetect.embedding import bertembedding
from SmartContractDetect.model.baseline.CNN.cnn import CNN_Model
from SmartContractDetect.model.baseline.CNNdataloader import NNDataset, GCN_Dataset
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from SmartContractDetect.model.baseline.GCN.gcn import GCN_Module

RE_train_csv = "../../../csvdata/contractData/RE_train_files.csv"
RE_test_csv = "../../../csvdata/contractData/RE_test_files.csv"
IO_train_csv = "../../../csvdata/contractData/IO_train_files.csv"
IO_test_csv = "../../../csvdata/contractData/IO_test_files.csv"
TO_train_csv = "../../../csvdata/contractData/TO_train_files.csv"
TO_test_csv = "../../../csvdata/contractData/TO_test_files.csv"


RE_train_dataset = GCN_Dataset(RE_train_csv)
RE_test_dataset = GCN_Dataset(RE_test_csv)
IO_train_dataset = GCN_Dataset(IO_train_csv)
IO_test_dataset = GCN_Dataset(IO_test_csv)
TO_train_dataset = GCN_Dataset(TO_train_csv)
TO_test_dataset = GCN_Dataset(TO_test_csv)

RE_train_dataloader = torch.utils.data.DataLoader(RE_train_dataset, batch_size=64, shuffle=True)
RE_test_dataloader = torch.utils.data.DataLoader(RE_test_dataset, batch_size=64, shuffle=True)
IO_train_dataloader = torch.utils.data.DataLoader(IO_train_dataset, batch_size=64, shuffle=True)
IO_test_dataloader = torch.utils.data.DataLoader(IO_test_dataset, batch_size=64, shuffle=True)
TO_train_dataloader = torch.utils.data.DataLoader(TO_train_dataset, batch_size=64, shuffle=True)
TO_test_dataloader = torch.utils.data.DataLoader(TO_test_dataset, batch_size=64, shuffle=True)

# with open("../../../train_RE.pkl", 'rb') as f:
#     RE_train_dataloader = pickle.load(f)
# with open("../../../test_RE.pkl", 'rb') as f:
#     RE_test_dataloader = pickle.load(f)
# with open("../../../train_IO.pkl", 'rb') as f:
#     IO_train_dataloader = pickle.load(f)
# with open("../../../test_IO.pkl", 'rb') as f:
#     IO_test_dataloader = pickle.load(f)
# with open("../../../train_TO.pkl", 'rb') as f:
#     TO_train_dataloader = pickle.load(f)
# with open("../../../test_TO.pkl", 'rb') as f:
#     TO_test_dataloader = pickle.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
model.to(device)


def test(test_dataloader, gnnmodel,vunl):
    checkpoint = torch.load(gnnmodel)
    gnnmodel = GCN_Module(768,512,1).to(device)
    gnnmodel.load_state_dict(checkpoint)
    gnnmodel.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_dataloader:
            labels = batch['label'].float().view(-1, 1).to(device)
            graphs = []
            for i in batch['graph']:
                G = read_dot(i)
                graphs.append(G)

            all_node_features = []
            all_adjacency_matrix = []

            # 遍历每个图进行处理
            for graph in graphs:
                nodes = graph.nodes(data=True)
                edges = graph.edges(data=True)
                node_features = {}
                edge_features = {}

                # 生成节点特征
                for node_id, data in nodes:
                    feature = data.get('label', '0')  # 假设节点特征存储在 'label' 字段
                    input = tokenizer(feature, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
                    with torch.no_grad():
                        outputs = model(**input)
                    feature = outputs.last_hidden_state[:, 0, :]
                    node_features[node_id] = torch.tensor(feature, dtype=torch.float32)

                # 生成邻接矩阵，边的存在用 1 表示，边的不存在用 0 表示
                num_nodes = len(graph.nodes())
                adj_matrix = torch.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵


                # 填充邻接矩阵
                for edge in edges:
                    u, v,data = edge
                    u_idx = list(graph.nodes()).index(u)  # 获取节点 u 的索引
                    v_idx = list(graph.nodes()).index(v)  # 获取节点 v 的索引
                    label = data.get('label', 0)  # 获取边的标签
                    if label is None:
                        adj_matrix[u_idx, v_idx] = 0
                    elif label == "True":
                        adj_matrix[u_idx, v_idx] = 1
                    else:
                        adj_matrix[u_idx, v_idx] = 2

                # 将每个图的节点特征和邻接矩阵加入到列表中
                all_node_features.append(node_features)
                all_adjacency_matrix.append(adj_matrix)
            node_feature_list = []
            for graphEmbedding, adj in zip(all_node_features, all_adjacency_matrix):
                node_ids = list(graphEmbedding.keys())
                node_feature = [graphEmbedding[node_id] for node_id in node_ids]
                node_feature = torch.stack(node_feature).to(device)
                node_feature = torch.squeeze(node_feature, 1)

                # 构建图数据对象
                data = Data(x=node_feature, edge_index=adj.nonzero().t().contiguous())

                node_feature_list.append(data)
            # 图变换（数据增强）
            #data_augmented = augmentations.apply_transforms(data)
            #node_feature_list.append(data_augmented)
            loader = torch_geometric.data.DataLoader(node_feature_list, batch_size=64,shuffle=False)
            graphembeddings =[]
            node_level_embeddings = []
            for batchA in loader:
                batchA = batchA.to(device)  # 将图数据移到GPU
                gat_embedding = gnnmodel(batchA.x, batchA.edge_index)
                print(gat_embedding.shape)
                batch = batchA.batch  # 获取每个节点所属的图
                num_graphs = batch.max().item() + 1  # 获取图的数量

                for i in range(num_graphs):
                    node_indices = (batch == i).nonzero(as_tuple=True)[0]
                    graphembedding = gat_embedding[node_indices]
                    node_level_embeddings.append(graphembedding)
                    graphembedding = graphembedding.mean(dim=0)
                    graphembeddings.append(graphembedding)

            graphembeddings = torch.stack(graphembeddings).to(device)
            preds = (graphembeddings > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    # Convert lists to numpy arrays for evaluation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    with open(vunl+ '_sc_' + 'result.txt', 'a') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("-" * 50 + "\n")
    # You can also return the metrics if needed for further processing
    return accuracy, precision, recall, f1

def train(train_dataloader, test_dataloader,vunl):
    num_epochs = 40
    patience = 3
    gnnmodel = GCN_Module(768,512,1).to(device)
    optimizer = optim.Adam(gnnmodel.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_loss = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(total=len(train_dataloader),desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_dataloader:
                labels = batch['label'].float().view(-1, 1).to(device)
                graphs = []
                for i in batch['graph']:
                    G = read_dot(i)
                    graphs.append(G)

                all_node_features = []
                all_adjacency_matrix = []

                # 遍历每个图进行处理
                for graph in graphs:
                    nodes = graph.nodes(data=True)
                    edges = graph.edges(data=True)
                    node_features = {}
                    edge_features = {}

                    # 生成节点特征
                    for node_id, data in nodes:
                        feature = data.get('label', '0')  # 假设节点特征存储在 'label' 字段
                        input = tokenizer(feature, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
                        with torch.no_grad():
                            outputs = model(**input)
                        feature = outputs.last_hidden_state[:, 0, :]
                        node_features[node_id] = torch.tensor(feature, dtype=torch.float32)

                    # 生成邻接矩阵，边的存在用 1 表示，边的不存在用 0 表示
                    num_nodes = len(graph.nodes())
                    adj_matrix = torch.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵


                    # 填充邻接矩阵
                    for edge in edges:
                        u, v,data = edge
                        u_idx = list(graph.nodes()).index(u)  # 获取节点 u 的索引
                        v_idx = list(graph.nodes()).index(v)  # 获取节点 v 的索引
                        label = data.get('label', 0)  # 获取边的标签
                        if label is None:
                            adj_matrix[u_idx, v_idx] = 0
                        elif label == "True":
                            adj_matrix[u_idx, v_idx] = 1
                        else:
                            adj_matrix[u_idx, v_idx] = 2

                    # 将每个图的节点特征和邻接矩阵加入到列表中
                    all_node_features.append(node_features)
                    all_adjacency_matrix.append(adj_matrix)
                node_feature_list = []
                for graphEmbedding, adj in zip(all_node_features, all_adjacency_matrix):
                    node_ids = list(graphEmbedding.keys())
                    node_feature = [graphEmbedding[node_id] for node_id in node_ids]
                    node_feature = torch.stack(node_feature).to(device)
                    node_feature = torch.squeeze(node_feature, 1)

                    # 构建图数据对象
                    data = Data(x=node_feature, edge_index=adj.nonzero().t().contiguous())

                    node_feature_list.append(data)
                # 图变换（数据增强）
                #data_augmented = augmentations.apply_transforms(data)
                #node_feature_list.append(data_augmented)
                loader = torch_geometric.data.DataLoader(node_feature_list, batch_size=64,shuffle=False)
                graphembeddings =[]
                node_level_embeddings = []
                for batchA in loader:
                    batchA = batchA.to(device)  # 将图数据移到GPU
                    gat_embedding = gnnmodel(batchA.x, batchA.edge_index)
                    print(gat_embedding.shape)
                    batch = batchA.batch  # 获取每个节点所属的图
                    num_graphs = batch.max().item() + 1  # 获取图的数量

                    for i in range(num_graphs):
                        node_indices = (batch == i).nonzero(as_tuple=True)[0]
                        graphembedding = gat_embedding[node_indices]
                        node_level_embeddings.append(graphembedding)
                        graphembedding = graphembedding.mean(dim=0)
                        graphembeddings.append(graphembedding)

                graphembeddings = torch.stack(graphembeddings).to(device)
                loss = nn.BCELoss()(graphembeddings, labels)
                epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            counter = 0
            best_loss = avg_loss
            torch.save(gnnmodel.state_dict(), vunl +'_sc_' +'' +'best_gnnmodel.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step(avg_loss)
    ##test
    test(test_dataloader, vunl+'_sc_'+''+'best_gnnmodel.pth',vunl)

train(TO_train_dataloader, TO_test_dataloader,'TO')
##train(RE_train_dataloader, RE_test_dataloader,'RE')
train(IO_train_dataloader, IO_test_dataloader,'IO')
