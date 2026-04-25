import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
import logging

from .spatial_attention import PKSAConv, DTWMaskCalculator, create_projection_matrix
from .temporal_attention import MAPTA, TraditionalTemporalAttention
from .embedding import DataEmbedding, TokenEmbedding


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatioTemporalGate(nn.Module):
    def __init__(self, d_model, gate_activation="sigmoid"):
        super().__init__()
        self.Wg1 = nn.Linear(d_model, d_model)
        self.Wg2 = nn.Linear(d_model, d_model)
        if gate_activation == "sigmoid":
            self.activation = torch.sigmoid
        elif gate_activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError("Unsupported gate activation function")

    def forward(self, H_spat, H_time):
        g = self.activation(self.Wg1(H_spat) + self.Wg2(H_time))
        H_fused = (1 - g) * H_spat + g * H_time
        return H_fused, g


class STEncoderBlock(nn.Module):
    def __init__(self, dim, s_attn_size, t_attn_size,
                 geo_num_heads=4, sem_num_heads=2, t_num_heads=2,
                 mlp_ratio=4., sem_attn_mode="dynamic", t_attn_mode="patch",
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 device=torch.device('cpu'), output_dim=1,
                 num_dgat_rounds=2, dtw_delta=5,
                 use_spatio_temporal_gate=True, fusion_method='gate',
                 gate_activation="sigmoid"):
        super().__init__()
        self.dim = dim
        self.sem_attn_mode = sem_attn_mode
        self.t_attn_mode = t_attn_mode
        self.use_spatio_temporal_gate = use_spatio_temporal_gate
        self.fusion_method = fusion_method

        self.norm_time = norm_layer(dim)

        spat_attn_out_channels = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.norm_spat = norm_layer(spat_attn_out_channels)

        if t_attn_mode == 'patch':
            self.time_attn = MAPTA(d_model=dim, d_t=dim // 2)
        elif t_attn_mode == 'traditional':
            self.time_attn = TraditionalTemporalAttention(dim, dim, t_num_heads)

        if sem_attn_mode == 'dynamic':
            self.spat_attn = PKSAConv(
                dim, spat_attn_out_channels,
                sem_num_heads, num_dgat_rounds=num_dgat_rounds,
                dtw_delta=dtw_delta
            )
            self.spat_proj = nn.Linear(spat_attn_out_channels, dim)

        if self.use_spatio_temporal_gate and self.fusion_method == 'gate':
            self.gate = SpatioTemporalGate(dim, gate_activation=gate_activation)
        elif self.fusion_method == 'concat':
            self.fusion_linear = nn.Linear(dim * 2, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_fusion = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, H_time_init=None, edge_index=None, M_theta=None, sem_mask=None):
        B, T, N, D = x.shape

        H_time_raw = self.time_attn(x)

        H_time = self.norm_time(H_time_raw)

        if self.sem_attn_mode == 'dynamic':
            H_spat_raw = self.spat_attn(x.permute(0, 2, 1, 3), edge_index, M_theta, tau=1.0, mask=sem_mask)
            logging.debug(f"kst_pan: H_spat_raw.shape = {H_spat_raw.shape}")
            H_spat_reshaped = H_spat_raw.permute(0, 2, 1, 3).reshape(B * N * T, -1)
            logging.debug(f"kst_pan: H_spat_reshaped.shape = {H_spat_reshaped.shape}")
            H_spat_norm = self.norm_spat(H_spat_reshaped)
            logging.debug(f"kst_pan: H_spat_norm.shape = {H_spat_norm.shape}")
            H_spat = H_spat_norm.reshape(B, N, T, -1).permute(0, 2, 1, 3)
            logging.debug(f"kst_pan: final H_spat.shape = {H_spat.shape}")
            H_spat = self.spat_proj(H_spat)
        else:
            H_spat = torch.zeros_like(x)

        if self.use_spatio_temporal_gate and self.fusion_method == 'gate':
            H_fused, gate_values = self.gate(H_spat, H_time)
        elif self.fusion_method == 'add':
            H_fused = H_spat + H_time
        elif self.fusion_method == 'concat':
            H_fused = self.fusion_linear(torch.cat([H_spat, H_time], dim=-1))
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        H_fused = x + self.drop_path(H_fused)
        H_fused = self.norm_fusion(H_fused)

        H_out = H_fused + self.drop_path(self.mlp(H_fused))

        return H_out


class KST_PAN(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self._scaler = data_feature.get('scaler')
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.ext_dim = data_feature.get("ext_dim", 0)
        self.num_batches = data_feature.get('num_batches', 1)
        self.dtw_matrix = data_feature.get('dtw_matrix')
        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')

        self.dataset = config.get('dataset', 'unknown')

        self.embed_dim = config.get('embed_dim', 64)
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)

        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)

        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 1)
        enc_depth = config.get("enc_depth", 6)

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)

        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)

        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)

        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.type_short_path = config.get("type_short_path", "hop")
        self.far_mask_delta = config.get('far_mask_delta', 5)
        self.dtw_delta = config.get('dtw_delta', 5)

        self.sem_attn_mode = config.get("sem_attn_mode", "dynamic")
        self.t_attn_mode = config.get("t_attn_mode", "patch")

        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)

        self.num_dgat_rounds = config.get("num_dgat_rounds", 2)
        self.use_spatio_temporal_gate = config.get("use_spatio_temporal_gate", True)
        self.fusion_method = config.get("fusion_method", "gate")
        self.gate_activation = config.get("gate_activation", "sigmoid")
        
        # Get activation and normalization layers from config
        act_layer_str = config.get('act_layer', 'GELU')
        norm_layer_str = config.get('norm_layer', 'LayerNorm')

        act_layer = getattr(nn, act_layer_str)
        if norm_layer_str == 'LayerNorm':
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            norm_layer = getattr(nn, norm_layer_str)


        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:
            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= self.far_mask_delta] = 1
            self.geo_mask = self.geo_mask.bool()

            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
            sem_mask_indices = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
            for i in range(self.sem_mask.shape[0]):
                self.sem_mask[i][sem_mask_indices[i]] = 0
            self.sem_mask = self.sem_mask.bool()

        self.pattern_keys = torch.from_numpy(data_feature.get('pattern_keys')).float().to(self.device)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx,
            drop=drop, add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week,
            device=self.device
        )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]

        # Convert adj_mx to Tensor and move to device to fix TypeError
        adj_mx = data_feature.get('adj_mx') # Get adj_mx from data_feature
        if adj_mx is None:
            raise ValueError("Adjacency matrix is required in data_feature for KST_PAN model initialization.")
        self.adj_mx = torch.from_numpy(adj_mx).float().to(self.device)
        self.edge_index = torch.where(self.adj_mx > 0)

        self.dtw_mask_calculator = DTWMaskCalculator(
            dtw_delta=self.dtw_delta,
            sigma_dist=1.0,
            device=self.device
        )
        M_theta = self.dtw_mask_calculator(
            self.adj_mx, sd_mx, self.dtw_matrix
        )
        self.register_buffer('M_theta', M_theta.to(self.device))

        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size,
                t_attn_size=self.t_attn_size,
                geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads,
                t_num_heads=t_num_heads, mlp_ratio=mlp_ratio,
                sem_attn_mode=self.sem_attn_mode, t_attn_mode=self.t_attn_mode,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=enc_dpr[i], act_layer=act_layer,
                norm_layer=norm_layer,
                device=self.device, output_dim=self.output_dim,
                num_dgat_rounds=self.num_dgat_rounds,
                dtw_delta=self.dtw_delta,
                use_spatio_temporal_gate=self.use_spatio_temporal_gate,
                fusion_method=self.fusion_method,
                gate_activation=self.gate_activation
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window,
            kernel_size=1, bias=True
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim,
            kernel_size=1, bias=True
        )

    def forward(self, batch, lap_mx=None):
        x = batch['X']
        T = x.shape[1]

        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)

        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))
        x_patterns = torch.cat(x_pattern_list, dim=-1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)

        enc = self.enc_embed_layer(x, lap_mx)

        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(
                enc,
                edge_index=self.edge_index,
                M_theta=self.M_theta,
                sem_mask=self.sem_mask
            )
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        # skip: (B, skip_dim, N, T) -> (B, T, N, skip_dim)
        skip = skip.permute(0, 3, 2, 1)
        
        # end_conv1 expects (B, input_window, N, skip_dim) -> (B, output_window, N, skip_dim)
        # where we treat T as channel
        skip = self.end_conv1(F.relu(skip))
        
        # For end_conv2, we need skip_dim as channel: (B, output_window, N, skip_dim) -> (B, skip_dim, N, output_window)
        skip = skip.permute(0, 3, 2, 1)
        skip = self.end_conv2(F.relu(skip))
        
        # Output: (B, output_dim, N, output_window) -> (B, output_window, N, output_dim)
        return skip.permute(0, 3, 2, 1)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)

    def get_data_feature(self):
        return {
            "scaler": self._scaler,
            "adj_mx": self.adj_mx,
            "num_nodes": self.num_nodes,
            "feature_dim": self.feature_dim,
            "ext_dim": self.ext_dim,
            "output_dim": self.output_dim,
            "num_batches": self.num_batches
        }
