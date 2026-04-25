import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim, lape_dim, adj_mx,
                 drop=0., add_time_in_day=False, add_day_in_week=False,
                 device=torch.device('cpu')):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim

        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)

        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)

        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)

        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx):
        origin_x = x

        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x = x + self.position_encoding(x)

        if self.add_time_in_day:
            x = x + self.daytime_embedding(
                (origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long()
            )

        if self.add_day_in_week:
            x = x + self.weekday_embedding(
                origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3)
            )

        x = x + self.spatial_embedding(lap_mx)
        x = self.dropout(x)
        return x
