import re

import pandas as pd
from torch.utils.data import Dataset

def preprocess_solidity_code(code):
    # 去掉注释
    code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.MULTILINE | re.DOTALL)

    # 去掉字符串字面量（可选，视具体需求而定）
    code = re.sub(r'"(.*?)"|\'(.*?)\'', '', code)

    # 去掉多余的空白字符
    code = re.sub(r'\s+', ' ', code).strip()

    return code

class NNDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.label = []
        df = pd.read_csv(csv_file).dropna()
        for index, row in df.iterrows():
            self.data.append(row['code'])
            self.label.append(row['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = int(self.label[idx])
        return {
            'text': data,
            'label': label
        }
class GCN_Dataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.label = []
        df = pd.read_csv(csv_file).dropna()
        for index, row in df.iterrows():
            self.data.append(row['graph'])
            self.label.append(row['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = int(self.label[idx])
        return {
            'graph': data,
            'label': label
        }