# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class TransformerModalFusionModel(nn.Module):
#     def __init__(self, text_embedding_dim, graph_embedding_dim, num_heads=8, num_layers=2, fusion_method='attention', mlp_hidden_dim=256):
#         super(TransformerModalFusionModel, self).__init__()
#
#         # 模态嵌入映射层
#         self.text_projection = nn.Linear(text_embedding_dim, 512)
#         self.graph_projection = nn.Linear(graph_embedding_dim, 512)
#
#         # 融合方法: 支持 'concat', 'add', 'attention' 等
#         self.fusion_method = fusion_method
#         if self.fusion_method == 'attention':
#             self.attention_weights = nn.Parameter(torch.rand(512))
#
#         # Transformer编码器
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=512, nhead=num_heads),
#             num_layers=num_layers
#         )
#
#         # 池化层
#         self.pooling = nn.AdaptiveAvgPool1d(withCL)
#
#         # 使用MLP替换原本的fc
#         self.mlp = nn.Sequential(
#             nn.Linear(512, mlp_hidden_dim),  # 第一层
#             nn.ReLU(),                       # 激活函数
#             nn.Linear(mlp_hidden_dim, withCL),
#             nn.ReLU(),
#             nn.Sigmoid()                     # 输出层，输出一个值用于二分类
#         )
#
#     def forward(self, text_embedding, graph_embedding):
#         # 嵌入映射
#         text_emb = self.text_projection(text_embedding)  # [batch_size, 512]
#         graph_emb = self.graph_projection(graph_embedding)  # [batch_size, 512]
#
#         # 融合方式处理
#         if self.fusion_method == 'concat':
#             combined = torch.cat((text_emb.unsqueeze(withCL), graph_emb.unsqueeze(withCL)), dim=withCL)  # [batch_size, 2, 512]
#         elif self.fusion_method == 'add':
#             combined = text_emb + graph_emb  # [batch_size, 512]
#             combined = combined.unsqueeze(withCL)  # [batch_size, withCL, 512]
#         elif self.fusion_method == 'attention':
#             combined = torch.stack((text_emb, graph_emb), dim=withCL)  # [batch_size, 2, 512]
#             attn_weights = F.softmax(self.attention_weights, dim=0)  # [512]
#             combined = combined * attn_weights  # [batch_size, 2, 512]
#
#         combined = combined.permute(withCL, 0, 2)  # [seq_len, batch_size, 512]
#
#         # Transformer编码
#         transformer_output = self.transformer_encoder(combined)  # [seq_len, batch_size, 512]
#
#         # 池化
#         pooled_output = self.pooling(transformer_output.permute(withCL, 2, 0)).squeeze(-withCL)  # [batch_size, 512]
#
#         # 使用MLP分类
#         output = self.mlp(pooled_output)  # 使用MLP进行分类
#
#         return output  # 输出 [batch_size, withCL] 的值，用于二分类
import torch
import torch.nn as nn
import torch.nn.functional as F




import torch
import torch.nn as nn

class LSTMTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMTextModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 512)  # 输出维度与图嵌入维度一致

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class TransformerModalFusionModel(nn.Module):
    def __init__(self, text_embedding_dim, graph_embedding_dim, lstm_hidden_dim=256, num_heads=8, num_layers=2, fusion_method='weighted', mlp_hidden_dim=256):
        super(TransformerModalFusionModel, self).__init__()

        # 模态嵌入映射层
        self.graph_projection = nn.Linear(graph_embedding_dim, 512)

        # LSTM文本模型
        self.lstm_text_model = LSTMTextModel(text_embedding_dim, lstm_hidden_dim)

        # 使用MLP进行最终分类
        self.mlp1 = nn.Sequential(
            nn.Linear(512, 1),  # 第一层
            # 输出层，输出一个值用于二分类
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 1),  # 第一层
            # 输出层，输出一个值用于二分类
        )
        self.sig = nn.Sigmoid()

        # 加权系数，用于加权融合
        self.text_weight = nn.Parameter(torch.tensor(0.8))
        self.graph_weight = nn.Parameter(torch.tensor(0.2))

        # 可选：设定融合方法
        self.fusion_method = fusion_method

    def forward(self, text_embedding, graph_embedding, labels):
        # 通过LSTM处理文本信息
        text_emb = self.lstm_text_model(text_embedding)  # [batch_size, 512]

        # 图嵌入映射
        graph_emb = self.graph_projection(graph_embedding)  # [batch_size, 512]

        # 进行预测
        text_pred = self.mlp1(text_emb)  # [batch_size, 1] 二分类预测
        graph_pred = self.mlp2(graph_emb)  # [batch_size, 1] 二分类预测

        # 决策融合: 进行加权平均或投票
        if self.fusion_method == 'weighted':
            # 使用加权平均
            final_pred = self.text_weight * text_pred + self.graph_weight * graph_pred  # [batch_size, 1]
        elif self.fusion_method == 'vote':
            # 使用投票（取最大值）
            final_pred = torch.max(text_pred, graph_pred)  # [batch_size, 1]

        return self.sig(final_pred)  # 输出最终预测值 [batch_size, 1]
