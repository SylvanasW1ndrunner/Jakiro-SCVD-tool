import torch
import torch.nn as nn

class SourceEncoder(nn.Module):
    def __init__(self, fasttext_model):  # 接收 FastTextModel 实例
        super(SourceEncoder, self).__init__()
        self.fasttext_model = fasttext_model  # 存储 FastTextModel 实例

    def forward(self, texts):
        # 假设 `texts` 是一个包含多个代码段或句子的批次
        # 使用 FastText 生成句子的向量
        text_embs = torch.stack([torch.tensor(self.fasttext_model.get_sentence_vector(text), dtype=torch.float32, requires_grad=True) for text in texts])
        return text_embs