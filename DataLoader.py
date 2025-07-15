import json
import os
import pickle
import re

import torch
import pandas as pd
from numpy.fft.helper import fftfreq
from torch.nn.functional import embedding
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    def __init__(self, vunl_csv, safe_csv,vunl_kind):
        self.data = []
        self.graph = []
        self.label = []
        self.files = []
        df1 = pd.read_csv(vunl_csv).dropna()
        df2 = pd.read_csv(safe_csv).dropna()
        df2 = df2.sample(n = int(len(df1) * 1.5),ignore_index=True)
        for index, row in df1.iterrows():
            if row['expression'] is None:
                print("NOne!!!!!!!!!!!!")
                continue
            self.data.append(row['expression'])
            self.graph.append(row['graph'])
            self.files.append(row['file'])
            if vunl_kind == 'RE':
                self.label.append(row['RE'])
            elif vunl_kind == 'IO':
                self.label.append(row['IO'])
            elif vunl_kind == 'TO':
                self.label.append(row['TO'])
        for index, row in df2.iterrows():
            if row['expression'] is None:
                print("NOne!!!!!!!!!!!!")
                continue
            self.data.append(row['expression'])
            self.graph.append(row['graph'])
            self.files.append(row['file'])
            self.label.append("0")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        graph = self.graph[idx]
        label = int(self.label[idx])

        # 删除中文标点符号
        return {
            'filename': self.files[idx],
            'text': data,
            'graph': graph,
            'label': label
        }
