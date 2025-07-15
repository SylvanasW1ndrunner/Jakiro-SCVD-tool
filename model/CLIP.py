import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import graph


class ContrastiveLearningModel(nn.Module):
    def __init__(self, text_embedding_dim, graph_embedding_dim,common_dim):
        super(ContrastiveLearningModel, self).__init__()
        self.text_encoder = nn.Linear(text_embedding_dim, common_dim)
        self.image_encoder = nn.Linear(graph_embedding_dim, common_dim)
        self.trans = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
        )
    def forward(self, text_embeddings, image_embeddings,masks):
        # 使用编码器处理输入模态
        text_features = self.text_encoder(text_embeddings)
        image_features = self.image_encoder(image_embeddings)
        print(text_features.shape,image_features.shape)
        # Normalize features to make cosine similarity computation more stable
        trans_output = self.trans(src=text_features.permute(1, 0, 2), tgt=image_features.permute(1, 0, 2), tgt_key_padding_mask = masks)
        trans_output2 = self.trans(src=image_features.permute(1, 0, 2), tgt=text_features.permute(1, 0, 2), src_key_padding_mask = masks)
        loss1 = self.compute_cosine_similarity_loss(trans_output.permute(1, 0, 2), image_embeddings,masks)
        trans_output2 = trans_output2.permute(1, 0, 2)
        text_features = text_features.mean(dim=1)
        trans_output2 = trans_output2.mean(dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        trans_output2 = F.normalize(trans_output2, p=2, dim=1)
        cos = (text_features * trans_output2).sum(dim=1)
        loss2 = 1-cos.mean()
        return (loss1 + loss2) / 2

    def compute_cosine_similarity_loss(self, transformed_graph, original_graph, masks):
        """
        使用余弦相似度进行逐节点对比，并根据掩码处理填充节点
        :param transformed_graph: Transformer 输出的图嵌入, 形状 [batch_size, node_nums, feature_dim]
        :param original_graph: 原始的图嵌入, 形状 [batch_size, node_nums, feature_dim]
        :param masks: 掩码张量，形状 [batch_size, node_nums]，1表示有效节点，0表示填充节点
        :return: 计算出的损失值
        """
        # 归一化每个节点的嵌入（L2范数归一化）
        transformed_graph = F.normalize(transformed_graph, p=2, dim=-1)  # [batch_size, node_nums, feature_dim]
        original_graph = F.normalize(original_graph, p=2, dim=-1)  # [batch_size, node_nums, feature_dim]

        # 逐节点余弦相似度
        cosine_similarity = (transformed_graph * original_graph).sum(dim=-1)  # [batch_size, node_nums]

        # 使用掩码忽略填充节点
        masked_cosine_similarity = cosine_similarity * masks  # [batch_size, node_nums]

        # 计算加权平均相似度，忽略填充节点
        masked_mean_similarity = masked_cosine_similarity.sum(dim=-1) / masks.sum(dim=-1)  # [batch_size]

        # 损失是负的平均相似度
        loss = 1 - masked_mean_similarity.mean()  # 对所有样本求平均损失
        return loss
