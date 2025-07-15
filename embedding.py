import json
import re
import pandas as pd
import requests
import scipy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, normalize, OneHotEncoder
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def bertembedding(texts, bert_model,tokenizer):
    inputs = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(**inputs)

    cls_vectors = outputs.last_hidden_state[:, 0,:]
    text_vectors = outputs.last_hidden_state
    return text_vectors,cls_vectors

def compute_kernel_bias(vecs, n_components=512):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    scaler = StandardScaler()
    vecs = scaler.fit_transform(vecs.to("cpu").numpy())
    cov = np.cov(vecs.T)
    u, s, vh = scipy.linalg.svd(cov, full_matrices=False)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    """ 最终向量标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def llmembedding(texts,model,tokenizer):
    inputs = tokenizer(texts, return_tensors='pt',padding=True, truncation=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    text_vectors = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    # print(text_vectors)
    kernel,bias = compute_kernel_bias(text_vectors)
    text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    text_vectors = normalize(text_vectors, norm='l2')
    # print(text_vectors)
    # kernel,bias = compute_kernel_bias(text_vectors)
    # text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    del inputs,outputs
    return text_vectors

# import torch
#
# # 检查CUDA是否可用
# cuda_available = torch.cuda.is_available()
#
# if cuda_available:
#     print("CUDA is available!")
#     # 获取CUDA设备数量
#     device_count = torch.cuda.device_count()
#     print(f"Device count: {device_count}")
#     # 获取CUDA设备名称
#     device_name = torch.cuda.get_device_name(0)  # 通常为0
#     print(f"Device name: {device_name}")
# else:
#     print("CUDA is not available.")



# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # 启用 int8 量化
#     llm_int8_threshold=6.0,  # 设置阈值控制哪些层应被量化
#     llm_int8_skip_modules=[],  # 可选择跳过某些模块的量化
# )

def infer_reentrancy(code):
    print(code)
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model":"llama3",
        "stream": False,
        "prompt": f"Does the following smart contract code contain a reentrancy vulnerability? Only respond with the number 1 or 0, with 1 indicating a vulnerability and 0 indicating no vulnerability.Here is the code：\n\n{code}\n\n",
        "options": {"temperature": 0.0, "stop": ["\n"]},
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response)
    res =response.json()
    return res.get('response','')

def remove_comments(solidity_code):
    # 删除单行注释
    no_single_comments = re.sub(r'//.*', '', solidity_code)

    # 删除多行注释
    no_comments = re.sub(r'/\*.*?\*/', '', no_single_comments, flags=re.DOTALL)

    return no_comments

import requests
import json
import numpy as np

# def get_embedding(batch):
#     embeddings = []  # 用于存储所有元素的embedding
#     print(f"Batch size: {len(batch)}")  # 打印批次大小
#
#     url = "http://localhost:11434/api/embeddings"
#     headers = {"Content-Type": "application/json"}
#
#     # 遍历批次中的每个元素
#     for code in batch:
#         payload = {
#             "model": "llama3",
#             "stream": False,
#             "prompt": f"{code}",
#         }
#         response = requests.post(url, headers=headers, data=json.dumps(payload))
#         res = response.json()
#
#         # 获取 'embedding' 部分
#         embedding = res.get('embedding', [])
#
#
#         # 如果 embedding 是一个 list，转换为 NumPy 数组
#         if isinstance(embedding, list):
#             embedding = np.array(embedding)
#
#         # 将每个元素的embedding添加到列表中
#         embeddings.append(embedding)
#
#     # 将所有embedding转换为NumPy数组，确保统一格式
#     embeddings = np.array(embeddings)
#     ##embeddings = torch.tensor(embeddings, dtype=torch.float32)
#     print(embeddings)
#     return embeddings



def graph_embedding(model, tokenizer, graphs):
    all_node_features = []  # 用于存储所有图的节点特征
    all_adjacency_matrices = []  # 用于存储所有图的邻接矩阵
    for graph in graphs:  # 遍历每个图
        nodes = graph.nodes(data=True)
        edges = graph.edges(data=True)
        node_features = {}
        edge_features = {}

        # 计算节点嵌入
        for node_id, data in nodes:
            feature = data['feature']
            inputs = tokenizer(feature, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # 获取节点的嵌入向量
            with torch.no_grad():

                outputs = model(**inputs)

                node_embedding = outputs.last_hidden_state[:, 0, :]

            node_features[node_id] = node_embedding

        # 计算边的特征
        colors = [data['color'] for _, _, data in edges]
        onehot_encoder = OneHotEncoder()
        color_encoded = onehot_encoder.fit_transform(np.array(colors).reshape(-1, 1)).toarray()

        for i, (source, target, _) in enumerate(edges):
            edge_features[(source, target)] = color_encoded[i]

        edge_colors = list(set([color for _, _, color in graph.edges(data='color')]))
        color_map = {color: i for i, color in enumerate(edge_colors)}
        adjacency_matrix = torch.zeros((len(graph), len(graph)))  # 初始化邻接矩阵

        # 更新邻接矩阵
        for source, target, color in graph.edges(data='color'):
            src_idx = list(graph.nodes()).index(source)
            tgt_idx = list(graph.nodes()).index(target)
            adjacency_matrix[src_idx, tgt_idx] = color_map[color]

        # 将当前图的特征和邻接矩阵添加到列表中
        all_node_features.append(node_features)
        all_adjacency_matrices.append(adjacency_matrix)
    return all_node_features, all_adjacency_matrices

