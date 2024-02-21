import torch
import torch.nn.functional as F
import torch.nn as nn

class SingleTransformerLayer(nn.Module):
    def __init__(self, embed_size, heads):
        super(SingleTransformerLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Self-Attention
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Output linear layer
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()
        # Split the embedding into heads
        x = x.view(batch_size, seq_len, self.heads, self.head_dim)

        # Self-Attention
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Scaled Dot-Product Attention
        energy = torch.einsum("nshd,nthd->nsthd", [queries, keys])  # batch_size, num_heads, seq_len, seq_len
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nsthd,nthd->nshd", [attention, values])
        out = out.reshape(batch_size, seq_len, self.heads * self.head_dim)

        # Output linear layer
        out = self.fc_out(out)
        return out

# 使用例
embed_size = 256
heads = 8
seq_len = 10
batch_size = 32

# ランダムな入力テンソルを生成
x = torch.rand(batch_size, seq_len, embed_size)

# トランスフォーマレイヤを作成
transformer_layer = SingleTransformerLayer(embed_size, heads)

# 入力をトランスフォーマレイヤに渡す
output = transformer_layer(x)

# 出力の形状を確認
print(output.shape)
