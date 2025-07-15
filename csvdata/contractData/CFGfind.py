import os

import pandas as pd
import pygraphviz as pgv
import networkx as nx

import pydot

def read_dot_with_subgraphs(path, name):
    # 使用 pydot 读取 dot 文件
    graphs = pydot.graph_from_dot_file(path)

    # 遍历图中的子图
    for graph in graphs:
        # 获取所有的子图
        subgraphs = graph.get_subgraphs()

        for subgraph in subgraphs:
            # 获取子图的名称并检查是否包含给定的名称
            if subgraph.get_name() and name in subgraph.get_name():
                dotstring = subgraph.to_string()
                return dotstring
    return None

def findCFG(csv):
    # 读取 CSV 文件
    df = pd.read_csv(csv)
    filelist = []
    namelist = []
    dotlist = []

    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        filename = str(row['file']).replace('.sol', '.dot')
        path = os.path.join(r'C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\finaldata\binarycode', filename)
        filelist.append(path)
        namelist.append(row['contract'])

    # 读取每个 DOT 文件并提取子图
    for i in range(len(filelist)):
        print(filelist[i])
        print(i)
        dotstring = read_dot_with_subgraphs(filelist[i], namelist[i])
        dotlist.append(dotstring)

    # 将 dotlist 添加为新的一列
    df['graph'] = dotlist

    # 将更新后的 DataFrame 写入 CSV 文件 # 生成新的 CSV 文件名
    df.to_csv(csv, index=False)  # 不写入行索引

    return csv

print(findCFG('test/RE_test.csv'))
print(findCFG('test/SAFE_test.csv'))
print(findCFG('train/RE_train.csv'))
findCFG('train/SAFE_train.csv')
findCFG('test/IO_test.csv')
findCFG('train/IO_train.csv')
findCFG('test/TO_test.csv')
findCFG('train/TO_train.csv')


