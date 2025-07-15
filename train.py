import os
import pickle
from distutils.command.install_egg_info import safe_name
import re
import numpy as np
import pandas as pd
import torch
import torch_geometric
from eth_utils.functional import combine
from matplotlib import pyplot as plt
from numpy.lib.format import open_memmap
from ply.yacc import token
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sympy.multipledispatch.dispatcher import source
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from tqdm import tqdm  # 导入tqdm
from visualdl import LogWriter
from SmartContractDetect.CFGGen import CFGGen, parse_feature
from SmartContractDetect.DataLoader import CustomDataset
from SmartContractDetect.embedding import  graph_embedding, bertembedding
from SmartContractDetect.model.CLIP import ContrastiveLearningModel
from SmartContractDetect.model.GAT import GraphSAGE
from SmartContractDetect.model.floss import FocalLoss
from SmartContractDetect.model.SourceEncoder import SourceEncoder
from SmartContractDetect.model.classifier import TransformerModalFusionModel
import pandas as pd



log_writer = LogWriter(logdir="log")

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 64
if os.path.exists('train_TO.pkl') == False and os.path.exists('test_TO.pkl') == False:
    re_train_csv = "csvdata/train/TO_train.csv"
    re_test_csv = "csvdata/test/TO_test.csv"
    safe_train_csv = "csvdata/train/SAFE_train.csv"
    safe_test_csv = "csvdata/test/SAFE_test.csv"
    TOtrain = CustomDataset(re_train_csv,safe_train_csv,"TO")
    TOtest = CustomDataset(re_test_csv,safe_test_csv,"TO")
    train_dataloader = DataLoader(TOtrain, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(TOtest, batch_size=batchsize, shuffle=True)
    with open('train_TO.pkl', 'wb') as f:
        pickle.dump(train_dataloader, f)
    with open('test_TO.pkl', 'wb') as f:
        pickle.dump(test_dataloader, f)
else:
    train_dataloader = pickle.load(open('train_TO.pkl', 'rb'))
    test_dataloader = pickle.load(open('test_TO.pkl', 'rb'))
print("train_dataloader",len(train_dataloader))
print("test_dataloader",len(test_dataloader))


def train():
    num_epochs = 20
    patience = 5  # 监测的epoch数
    print("start load!!")
    model = RobertaModel.from_pretrained(r"microsoft/codebert-base")
    tokenizer = RobertaTokenizer.from_pretrained(r"microsoft/codebert-base")    # 模型下载
    model.to(device)
    gat_model = GraphSAGE(nfeat=768, nhid=512, nclass=512, dropout=0.1).to(device)
    contrastive_model = ContrastiveLearningModel(text_embedding_dim=768, graph_embedding_dim=512,common_dim=512).to(device)
    #classifier_model = StackClassifier(text_embedding_dim=4096, graph_embedding_dim=512, num_heads=8, num_layers=2).to(device)
    classifier_model = TransformerModalFusionModel(text_embedding_dim=768, graph_embedding_dim=512, num_heads=8, num_layers=2).to(device)
    floss = FocalLoss()
   # augmentations = GraphAugmentations(drop_prob=1, mask_prob=0.2, edge_perturb_prob=0.2)
    optimizer = torch.optim.Adam(
        list(gat_model.parameters()) +
        ##list(contrastive_model.parameters()) +
        list(classifier_model.parameters()),
        lr=0.001
    )

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_loss = float('inf')
    counter = 0
    avgepoch_loss = []
    epoch_acc = []
    epoch_f1 = []
    epoch_call = []
    for epoch in range(num_epochs):
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []
        epoch_loss = 0
        source_embeddings_list = []
        graph_embeddings_list = []
        labels_list = []
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_dataloader:
                # 将数据移到GPU
                labels = batch['label'].float().view(-1, 1).to(device)
                ##source_embedding = get_embedding(batch['text']).to(device)
                source_embedding,cls_embeddings = bertembedding(batch['text'],model,tokenizer)
                source_embedding = torch.tensor(source_embedding,dtype=torch.float32).to(device)
                cls_embeddings = torch.tensor(cls_embeddings,dtype=torch.float32).to(device)
                graphs = []
                for i in batch['graph']:
                    G = CFGGen(i)
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
                        feature = data['label']  # 假设节点特征存储在 'label' 字段
                        feature = parse_feature(feature)
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
                loader = torch_geometric.data.DataLoader(node_feature_list, batch_size=batchsize,shuffle=False)
                graphembeddings =[]
                node_level_embeddings = []
                for batchA in loader:
                    batchA = batchA.to(device)  # 将图数据移到GPU
                    gat_embedding = gat_model(batchA.x, batchA.edge_index)
                    print(gat_embedding.shape)
                    batch = batchA.batch  # 获取每个节点所属的图
                    num_graphs = batch.max().item() + 1  # 获取图的数量

                    for i in range(num_graphs):
                        node_indices = (batch == i).nonzero(as_tuple=True)[0]
                        graphembedding = gat_embedding[node_indices]
                        node_level_embeddings.append(graphembedding)
                        graphembedding = graphembedding.mean(dim=0)
                        graphembeddings.append(graphembedding)
                # 获取最大节点数
                max_nodes = max([graph.size(0) for graph in node_level_embeddings])
                print("max_nodes:",max_nodes)
                # 创建一个掩码张量
                masks = []
                padded_embeddings = []
                for graph_embedding in node_level_embeddings:
                    pad_size = max_nodes - graph_embedding.size(0)
                    if pad_size > 0:
                        # 创建掩码：1 表示有效节点，0 表示填充节点
                        mask = torch.cat([torch.ones(graph_embedding.size(0)), torch.zeros(pad_size)]).to(device)
                        # 填充嵌入
                        padding = torch.zeros(pad_size, graph_embedding.size(1)).to(device)
                        padded_graph_embedding = torch.cat([graph_embedding, padding], dim=0)
                    else:
                        mask = torch.ones(graph_embedding.size(0)).to(device)
                        padded_graph_embedding = graph_embedding
                    padded_embeddings.append(padded_graph_embedding)
                    masks.append(mask)

                padded_embeddings = torch.stack(padded_embeddings).to(device)
                masks = torch.stack(masks).to(device)
                graphembeddings = torch.stack(graphembeddings).to(device)
                    # 前向传播到对比学习模型
                cl_loss = contrastive_model(source_embedding, padded_embeddings,masks)
                labels_list.append(labels.detach().cpu())
                    # 计算InfoNCE损失

                pre = classifier_model(source_embedding, graphembeddings,labels.to(device))
                print(f'Predictions shape: {pre.shape}, Labels shape: {labels.shape}')
                ##fusion_loss = floss(pre, labels)
                fusion_loss = nn.BCELoss()(pre, labels)
                    #loss = loss_1 + laaoss_2
                loss = fusion_loss + cl_loss * (epoch / num_epochs)
                ##loss = fusion_loss
                epoch_loss += loss.item()
                preds = (pre > 0.5).float()
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)

                all_labels.append(labels.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())
                    # 反向传播和优化步骤
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ##pbar.set_postfix(loss_total=loss.item(), loss_clip=cl_loss.item(), loss_bce=fusion_loss.item())
                pbar.set_postfix(loss_total=loss.item(), loss_bce=fusion_loss.item(),loss_cl = cl_loss.item())
                ##pbar.set_postfix(loss_total=loss.item(), loss_bce=fusion_loss.item())
                pbar.update(1)
            scheduler.step(epoch_loss)
            avg_loss = epoch_loss / len(train_dataloader)
            print(f'Epoch: {epoch}, Avg Loss: {avg_loss}')
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions * 100

        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        print(f'Epoch: {epoch}, Avg Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        avgepoch_loss.append(avg_loss)
        epoch_acc.append(accuracy)
        epoch_f1.append(f1)
        epoch_call.append(recall)
        # 保存模型
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'gat_model_state_dict': gat_model.state_dict(),
                'contrastive_model_state_dict': contrastive_model.state_dict(),
                'classifier_model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'funcModelMultiCon/TO/withCL/finalmodel/model_{epoch}.pth')

        # 检查是否需要提前停止
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'gat_model_state_dict': gat_model.state_dict(),
                'contrastive_model_state_dict': contrastive_model.state_dict(),
                'classifier_model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'funcModelMultiCon/TO/withCL/finalmodel/best_model_80.pth')
        else:
            counter += 1  # 只有在损失未下降时才增加计数
            if counter >= patience:
                print("Early stopping triggered")
                break

    # 在训练结束后保存最后的模型
    torch.save({
        'epoch': epoch,
        'gat_model_state_dict': gat_model.state_dict(),
        'contrastive_model_state_dict': contrastive_model.state_dict(),
        'classifier_model_state_dict': classifier_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, 'funcModelMultiCon/TO/withCL/finalmodel/last_model_80.pth')
    with open('funcModelMultiCon/TO/withCL/finalresult/train_loss.txt', 'w') as f:
        for loss, acc, f1, recall in zip(avgepoch_loss, epoch_acc, epoch_f1, epoch_call):
            f.write(f'loss:{loss}, acc:{acc}, f1:{f1}, recall:{recall}\n')

def test(model_path, report_path='test_report.txt'):
    # 加载最佳模型
    model = RobertaModel.from_pretrained(r"microsoft/codebert-base")
    tokenizer = RobertaTokenizer.from_pretrained(r"microsoft/codebert-base")
    model.to(device)
    gat_model = GraphSAGE(nfeat=768, nhid=512, nclass=512, dropout=0.1).to(device)
    contrastive_model = ContrastiveLearningModel(text_embedding_dim=768, graph_embedding_dim=512, common_dim=512).to(device)
    classifier_model = TransformerModalFusionModel(text_embedding_dim=768, graph_embedding_dim=512, num_heads=8, num_layers=2).to(device)
    # 加载训练时保存的模型
    checkpoint = torch.load(model_path)
    gat_model.load_state_dict(checkpoint['gat_model_state_dict'])
    contrastive_model.load_state_dict(checkpoint['contrastive_model_state_dict'],strict=False)
    classifier_model.load_state_dict(checkpoint['classifier_model_state_dict'])

    gat_model.eval()
    contrastive_model.eval()
    classifier_model.eval()

    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []
    all_probs = [] #  <-- 新增：用于存储模型的原始概率输出

    with torch.no_grad():
        for batch in test_dataloader:
            labels = batch['label'].float().view(-1, 1).to(device)
            source_embedding,cls_embeddings = bertembedding(batch['text'],model,tokenizer)
            source_embedding = torch.tensor(source_embedding,dtype=torch.float32).to(device)
            cls_embeddings = torch.tensor(cls_embeddings,dtype=torch.float32).to(device)
            graphs = []
            for i in batch['graph']:
                G = CFGGen(i)
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
                    feature = data['label']  # 假设节点特征存储在 'label' 字段
                    feature = parse_feature(feature)
                    input = tokenizer(feature, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
                    with torch.no_grad():
                        outputs = model(**input)
                    feature = outputs.last_hidden_state[:, 0, :]
                    node_features[node_id] = torch.tensor(feature, dtype=torch.float32)

                # 生成邻接矩阵
                num_nodes = len(graph.nodes())
                adj_matrix = torch.zeros((num_nodes, num_nodes))

                # 填充邻接矩阵
                for edge in edges:
                    u, v,data = edge
                    u_idx = list(graph.nodes()).index(u)
                    v_idx = list(graph.nodes()).index(v)
                    label = data.get('label', 0)
                    if label is None:
                        adj_matrix[u_idx, v_idx] = 0
                    elif label == "True":
                        adj_matrix[u_idx, v_idx] = 1
                    else:
                        adj_matrix[u_idx, v_idx] = 2
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

            loader = torch_geometric.data.DataLoader(node_feature_list, batch_size=batchsize,shuffle=False)
            graphembeddings = []
            node_level_embeddings = []
            for batchA in loader:
                batchA = batchA.to(device)
                gat_embedding = gat_model(batchA.x, batchA.edge_index)
                batch = batchA.batch
                num_graphs = batch.max().item() + 1

                for i in range(num_graphs):
                    node_indices = (batch == i).nonzero(as_tuple=True)[0]
                    graphembedding = gat_embedding[node_indices]
                    node_level_embeddings.append(graphembedding)
                    graphembedding = graphembedding.mean(dim=0)
                    graphembeddings.append(graphembedding)

            graphembeddings = torch.stack(graphembeddings).to(device)

            # 前向传播
            pre = classifier_model(source_embedding, graphembeddings, labels.to(device))

            # 计算损失
            loss = nn.BCELoss()(pre, labels)
            epoch_loss += loss.item()

            # 计算准确率
            preds = (pre > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probs.append(pre.cpu().numpy()) # <-- 新增：收集原始概率

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probs = np.concatenate(all_probs, axis=0) # <-- 新增：整合所有原始概率

    avg_loss = epoch_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions * 100

    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    # --- 新增：ROC曲线和AUC计算 ---
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    # --- ROC曲线和AUC计算结束 ---

    # 打印结果
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'AUC: {roc_auc:.4f}') # <-- 新增：打印AUC值

    # --- 新增：绘制并保存ROC曲线 ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 将ROC曲线图保存到文件
    roc_figure_path = report_path.replace('.txt', '_roc_curve.png')
    plt.savefig(roc_figure_path)
    print(f'ROC curve saved to {roc_figure_path}')
    plt.show()
    # --- 绘制结束 ---

    # 保存测试报告到本地文件
    with open(report_path, 'w') as f:
        f.write(f'Test Loss: {avg_loss:.4f}\n')
        f.write(f'Accuracy: {accuracy:.4f}%\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'AUC: {roc_auc:.4f}\n') # <-- 新增：将AUC写入报告

def test2(model_path, report_path='test_report.txt'):
    # 加载最佳模型
    model = AutoModel.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
    tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
    model.to(device)
    gat_model = GraphSAGE(nfeat=768, nhid=512, nclass=512, dropout=0.1).to(device)
    ##contrastive_model = ContrastiveLearningModel(text_embedding_dim=768, graph_embedding_dim=512, common_dim=512).to(device)
    classifier_model = TransformerModalFusionModel(text_embedding_dim=768, graph_embedding_dim=512, num_heads=8, num_layers=2).to(device)
    # 加载训练时保存的模型
    checkpoint = torch.load(model_path)
    gat_model.load_state_dict(checkpoint['gat_model_state_dict'])
    ##contrastive_model.load_state_dict(checkpoint['contrastive_model_state_dict'],strict=False)
    classifier_model.load_state_dict(checkpoint['classifier_model_state_dict'])

    gat_model.eval()
    ##contrastive_model.eval()
    classifier_model.eval()

    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            labels = batch['label'].float().view(-1, 1).to(device)
            source_embedding,cls_embeddings = bertembedding(batch['text'],model,tokenizer)
            source_embedding = torch.tensor(source_embedding,dtype=torch.float32).to(device)
            cls_embeddings = torch.tensor(cls_embeddings,dtype=torch.float32).to(device)
            graphs = []
            for i in batch['graph']:
                G = CFGGen(i)
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
                    feature = data['label']  # 假设节点特征存储在 'label' 字段
                    feature = parse_feature(feature)
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

            loader = torch_geometric.data.DataLoader(node_feature_list, batch_size=batchsize,shuffle=False)
            graphembeddings = []
            node_level_embeddings = []
            for batchA in loader:
                batchA = batchA.to(device)
                gat_embedding = gat_model(batchA.x, batchA.edge_index)
                batch = batchA.batch  # 获取每个节点所属的图
                num_graphs = batch.max().item() + 1  # 获取图的数量

                for i in range(num_graphs):
                    node_indices = (batch == i).nonzero(as_tuple=True)[0]
                    graphembedding = gat_embedding[node_indices]
                    node_level_embeddings.append(graphembedding)
                    graphembedding = graphembedding.mean(dim=0)
                    graphembeddings.append(graphembedding)
            max_nodes = max([graph.size(0) for graph in node_level_embeddings])

            # 创建一个掩码张量
            masks = []
            padded_embeddings = []
            for graph_embedding in node_level_embeddings:
                pad_size = max_nodes - graph_embedding.size(0)
                if pad_size > 0:
                    # 创建掩码：1 表示有效节点，0 表示填充节点
                    mask = torch.cat([torch.ones(graph_embedding.size(0)), torch.zeros(pad_size)]).to(device)
                    # 填充嵌入
                    padding = torch.zeros(pad_size, graph_embedding.size(1)).to(device)
                    padded_graph_embedding = torch.cat([graph_embedding, padding], dim=0)
                else:
                    mask = torch.ones(graph_embedding.size(0)).to(device)
                    padded_graph_embedding = graph_embedding
                padded_embeddings.append(padded_graph_embedding)
                masks.append(mask)

            padded_embeddings = torch.stack(padded_embeddings).to(device)
            masks = torch.stack(masks).to(device)
            graphembeddings = torch.stack(graphembeddings).to(device)

            # 前向传播
            pre = classifier_model(source_embedding, graphembeddings,labels.to(device))
            print(pre.shape,labels.shape)
            # 计算损失
            loss = nn.BCELoss()(pre,labels)
            epoch_loss += loss.item()

            # 计算准确率
            preds = (pre > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    avg_loss = epoch_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions * 100

    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    # 打印结果
    print(f'Test Loss: {avg_loss}, Accuracy: {accuracy:.2f}%')

    # 保存测试报告到本地文件
    with open(report_path, 'w') as f:
        f.write(f'Test Loss: {avg_loss}\n')
        f.write(f'Accuracy: {accuracy:.4f}%\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')

def train2():
    num_epochs = 10
    patience = 5  # 监测的epoch数
    print("start load!!")
    model = AutoModel.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
    tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
    # 模型下载
    model.to(device)
    gat_model = GraphSAGE(nfeat=768, nhid=512, nclass=512, dropout=0.1).to(device)
    contrastive_model = ContrastiveLearningModel(text_embedding_dim=768, graph_embedding_dim=512,common_dim=512).to(device)
    #classifier_model = StackClassifier(text_embedding_dim=4096, graph_embedding_dim=512, num_heads=8, num_layers=2).to(device)
    classifier_model = TransformerModalFusionModel(text_embedding_dim=768, graph_embedding_dim=512, num_heads=8, num_layers=2).to(device)
    floss = FocalLoss()
    # augmentations = GraphAugmentations(drop_prob=1, mask_prob=0.2, edge_perturb_prob=0.2)
    optimizer = torch.optim.Adam(
        list(gat_model.parameters()) +
        ##list(contrastive_model.parameters()) +
        list(classifier_model.parameters()),
        lr=0.001
    )

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_loss = float('inf')
    counter = 0
    avgepoch_loss = []
    epoch_acc = []
    epoch_f1 = []
    epoch_call = []
    for epoch in range(num_epochs):
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []
        epoch_loss = 0
        source_embeddings_list = []
        graph_embeddings_list = []
        labels_list = []
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_dataloader:
                # 将数据移到GPU
                labels = batch['label'].float().view(-1, 1).to(device)
                ##source_embedding = get_embedding(batch['text']).to(device)
                source_embedding,cls_embeddings = bertembedding(batch['text'],model,tokenizer)
                source_embedding = torch.tensor(source_embedding,dtype=torch.float32).to(device)
                cls_embeddings = torch.tensor(cls_embeddings,dtype=torch.float32).to(device)
                graphs = []
                for i in batch['graph']:
                    G = CFGGen(i)
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
                        feature = data['label']  # 假设节点特征存储在 'label' 字段
                        feature = parse_feature(feature)
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
                loader = torch_geometric.data.DataLoader(node_feature_list, batch_size=batchsize,shuffle=False)
                graphembeddings =[]
                node_level_embeddings = []
                for batchA in loader:
                    batchA = batchA.to(device)  # 将图数据移到GPU
                    gat_embedding = gat_model(batchA.x, batchA.edge_index)
                    print(gat_embedding.shape)
                    batch = batchA.batch  # 获取每个节点所属的图
                    num_graphs = batch.max().item() + 1  # 获取图的数量

                    for i in range(num_graphs):
                        node_indices = (batch == i).nonzero(as_tuple=True)[0]
                        graphembedding = gat_embedding[node_indices]
                        node_level_embeddings.append(graphembedding)
                        graphembedding = graphembedding.mean(dim=0)
                        graphembeddings.append(graphembedding)
                # 获取最大节点数
                max_nodes = max([graph.size(0) for graph in node_level_embeddings])

                # 创建一个掩码张量
                masks = []
                padded_embeddings = []
                for graph_embedding in node_level_embeddings:
                    pad_size = max_nodes - graph_embedding.size(0)
                    if pad_size > 0:
                        # 创建掩码：1 表示有效节点，0 表示填充节点
                        mask = torch.cat([torch.ones(graph_embedding.size(0)), torch.zeros(pad_size)]).to(device)
                        # 填充嵌入
                        padding = torch.zeros(pad_size, graph_embedding.size(1)).to(device)
                        padded_graph_embedding = torch.cat([graph_embedding, padding], dim=0)
                    else:
                        mask = torch.ones(graph_embedding.size(0)).to(device)
                        padded_graph_embedding = graph_embedding
                    padded_embeddings.append(padded_graph_embedding)
                    masks.append(mask)

                padded_embeddings = torch.stack(padded_embeddings).to(device)
                masks = torch.stack(masks).to(device)
                graphembeddings = torch.stack(graphembeddings).to(device)
                # 前向传播到对比学习模型
                # cl_loss = contrastive_model(source_embedding, padded_embeddings,masks)
                labels_list.append(labels.detach().cpu())
                # 计算InfoNCE损失

                pre = classifier_model(source_embedding, graphembeddings,labels.to(device))
                print(f'Predictions shape: {pre.shape}, Labels shape: {labels.shape}')
                ##fusion_loss = floss(pre, labels)
                fusion_loss = nn.BCELoss()(pre, labels)
                #loss = loss_1 + laaoss_2
                # loss = fusion_loss + cl_loss * (epoch / num_epochs)
                loss = fusion_loss
                epoch_loss += loss.item()
                preds = (pre > 0.5).float()
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)

                all_labels.append(labels.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())
                # 反向传播和优化步骤
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ##pbar.set_postfix(loss_total=loss.item(), loss_clip=cl_loss.item(), loss_bce=fusion_loss.item())
                ##pbar.set_postfix(loss_total=loss.item(), loss_bce=fusion_loss.item(),loss_cl = cl_loss.item())
                pbar.set_postfix(loss_total=loss.item(), loss_bce=fusion_loss.item())
                pbar.update(1)
            scheduler.step(epoch_loss)
            avg_loss = epoch_loss / len(train_dataloader)
            print(f'Epoch: {epoch}, Avg Loss: {avg_loss}')
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions * 100

        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        print(f'Epoch: {epoch}, Avg Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        avgepoch_loss.append(avg_loss)
        epoch_acc.append(accuracy)
        epoch_f1.append(f1)
        epoch_call.append(recall)
        # 保存模型
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'gat_model_state_dict': gat_model.state_dict(),
                'contrastive_model_state_dict': contrastive_model.state_dict(),
                'classifier_model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'funcModelMultiCon/TO/noCL/finalmodel/model_{epoch}.pth')

        # 检查是否需要提前停止
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'gat_model_state_dict': gat_model.state_dict(),
                'contrastive_model_state_dict': contrastive_model.state_dict(),
                'classifier_model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'funcModelMultiCon/TO/noCL/finalmodel/best_model_80.pth')
        else:
            counter += 1  # 只有在损失未下降时才增加计数
            if counter >= patience:
                print("Early stopping triggered")
                break

    # 在训练结束后保存最后的模型
    torch.save({
        'epoch': epoch,
        'gat_model_state_dict': gat_model.state_dict(),
        'contrastive_model_state_dict': contrastive_model.state_dict(),
        'classifier_model_state_dict': classifier_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, 'funcModelMultiCon/TO/noCL/finalmodel/last_model_80.pth')
    with open('funcModelMultiCon/TO/noCL/finalresult/train_loss.txt', 'w') as f:
        for loss, acc, f1, recall in zip(avgepoch_loss, epoch_acc, epoch_f1, epoch_call):
            f.write(f'loss:{loss}, acc:{acc}, f1:{f1}, recall:{recall}\n')


if __name__ == '__main__':
    train()
    test('funcModelMultiCon/TO/withCL/finalmodel/best_model_80.pth', report_path='funcModelMultiCon/TO/withCL/finalresult/test_report_80.txt')
    ##train2()