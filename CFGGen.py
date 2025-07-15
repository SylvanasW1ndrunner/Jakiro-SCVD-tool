import re

import networkx as nx
import pandas as pd
import pydot
import pygraphviz as pgv
from graphviz import Source


def process_dot_content(dot_content):
    """
    Process DOT content to convert subgraphs to digraphs and remove the label attribute.

    Args:
        dot_content (str): String containing DOT content.

    Returns:
        str: Processed DOT content.
    """
    try:
        # Replace "subgraph" with "digraph"
        dot_content = dot_content.replace("subgraph", "strict digraph", 1)

        # Remove the first occurrence of a label="..." pattern
        dot_content = re.sub(r'label=".*?";', '', dot_content, count=1)

        return dot_content
    except Exception as e:
        print(f"Error processing DOT content: {e}")
        return None

def CFGGen(data):
    graph = pgv.AGraph(string=data)

    # 使用 NetworkX 读取图
    G = nx.nx_agraph.from_agraph(graph)
    # 打印节点和边

    return G

def read_dot(data):
    data = process_dot_content(data)
    graph = pgv.AGraph(string=data)
    G =  nx.nx_agraph.from_agraph(graph)
    return G

def read_dot_with_subgraphs(path):
    # 使用 pygraphviz 读取 dot 文件
    graph = pgv.AGraph(path)

    # 使用 NetworkX 读取图结构
    G = nx.nx_agraph.from_agraph(graph)

    # 直接遍历 pygraphviz 图中的子图
    for subgraph in graph.subgraphs():
        print(f"Subgraph label: {subgraph.name}")

        # 将子图转换为 NetworkX 格式
        sub_G = nx.nx_agraph.from_agraph(subgraph)

        # 打印子图的节点和边
        print("Nodes in subgraph:")
        for node, data in sub_G.nodes(data=True):
            print(f"Node: {node}, Attributes: {data}")

        print("Edges in subgraph:")
        for edge in sub_G.edges(data=True):
            print(f"Edge: {edge}")

    return G

def parse_feature(feature):
    node_type_pattern = r"Node\sType:\s*(.*?)\s+\d*"
    node_types = re.findall(node_type_pattern, feature)

    # 提取 IRs（直到文本结尾或下一个 NodeType）
    irs_pattern = r'EXPRESSION:\s*(.*?)\s*IRs:'
    irs = re.findall(irs_pattern, feature)
    node_types = "".join(node_types).replace("\n", "")
    irs = "".join(irs).replace("\n", ";").strip()
    return irs


# RE_train_path = 'csvdata/train/RE_train.csv'
# SAFE_train_path = 'csvdata/train/SAFE_train.csv'
# df = pd.read_csv(RE_train_path)
#
# for index, row in df.iterrows():
#     G = CFGGen(row['graph'])
#     for idx,data in G.nodes(data=True):
#         feature = parse_feature(data['label'])
#         print(feature)
#     break