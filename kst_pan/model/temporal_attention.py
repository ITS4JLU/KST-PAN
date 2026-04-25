import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class DFTSeasonalExtractor(nn.Module):
    def __init__(self, d_model, k_freq=1):
        super().__init__()
        self.k_freq = k_freq
        self.d_model = d_model

    def forward(self, z):
        B, T, D = z.shape

        z_fft = torch.fft.rfft(z, dim=1)
        freq_magnitudes = torch.abs(z_fft)

        top_k_indices = torch.topk(freq_magnitudes, self.k_freq, dim=1)[1]

        mask = torch.zeros_like(freq_magnitudes)
        mask.scatter_(1, top_k_indices, freq_magnitudes.gather(1, top_k_indices))

        z_fft_filtered = z_fft * (mask / (freq_magnitudes + 1e-8))
        z_seasonal = torch.fft.irfft(z_fft_filtered, n=T, dim=1)

        return z_seasonal


class MultiScaleTrendExtractor(nn.Module):
    def __init__(self, d_model, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.n_k = len(kernel_sizes)

        self.query_proj = nn.Linear(d_model, self.n_k)

    def forward(self, z_rem):
        B, T, D = z_rem.shape

        alpha = self.query_proj(z_rem.mean(dim=1))
        alpha = F.softmax(alpha, dim=-1)

        pooled_results = []
        for k in self.kernel_sizes:
            if k > T:
                pooled = F.adaptive_avg_pool1d(z_rem.permute(0, 2, 1), T).permute(0, 2, 1)
            else:
                pooled = F.avg_pool1d(
                    z_rem.permute(0, 2, 1),
                    kernel_size=k,
                    stride=1,
                    padding=k // 2
                ).permute(0, 2, 1)
            pooled_results.append(pooled)

        pooled_stack = torch.stack(pooled_results, dim=1)

        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        z_trend = (pooled_stack * alpha).sum(dim=1)

        return z_trend


class NoisyGatingRouter(nn.Module):
    def __init__(self, d_model, k_route, total_patch_sizes):
        super().__init__()
        self.k_route = k_route
        self.total_patch_sizes = total_patch_sizes

        self.feature_proj = nn.Linear(d_model * 3, d_model)
        self.routing_linear = nn.Linear(d_model, total_patch_sizes)
        self.noise_linear = nn.Linear(d_model, total_patch_sizes)

    def forward(self, z_n, z_sea, z_trend, patch_candidates):
        z_concat = torch.cat([z_n, z_sea, z_trend], dim=-1)
        z_fused = self.feature_proj(z_concat)

        noise = torch.randn_like(z_fused) * 0.1
        gating_input = z_fused + noise

        routing_logits = self.routing_linear(gating_input)

        noise_scale = F.softplus(self.noise_linear(gating_input))
        routing_logits = routing_logits + torch.randn_like(routing_logits) * noise_scale

        routing_weights = F.softmax(routing_logits, dim=-1)

        top_k_values, top_k_indices = torch.topk(routing_weights, self.k_route, dim=-1)

        sparse_routing = torch.zeros_like(routing_weights)
        sparse_routing.scatter_(1, top_k_indices, top_k_values)

        selected_patches = patch_candidates[top_k_indices]

        return sparse_routing, selected_patches, top_k_indices


class LocalPatchAttention(nn.Module):
    def __init__(self, d_model, d_t):
        super().__init__()
        self.d_t = d_t
        self.q_proj = nn.Linear(d_model, d_t)
        self.k_proj = nn.Linear(d_model, d_t)
        self.v_proj = nn.Linear(d_model, d_t)
        self.scale = d_t ** -0.5

    def forward(self, patch):
        B, N_p, P, D = patch.shape

        q = self.q_proj(patch.mean(dim=2, keepdim=True))
        k = self.k_proj(patch)
        v = self.v_proj(patch)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).squeeze(2)

        return out


class GlobalPatchAttention(nn.Module):
    def __init__(self, d_model, d_t, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.d_t = d_t
        self.scale = d_t ** -0.5

        self.q_proj = nn.Linear(patch_size * d_model, d_t)
        self.k_proj = nn.Linear(patch_size * d_model, d_t)
        self.v_proj = nn.Linear(patch_size * d_model, d_t)

    def forward(self, flattened_patches):
        B, N_p, P_D = flattened_patches.shape

        q = self.q_proj(flattened_patches).unsqueeze(2)
        k = self.k_proj(flattened_patches).unsqueeze(1)
        v = self.v_proj(flattened_patches).unsqueeze(1)

        attn = torch.sigmoid(q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).squeeze(2)

        return out


class DualPatchAttention(nn.Module):
    def __init__(self, d_model, d_t, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.local_attn = LocalPatchAttention(d_model, d_t)
        self.global_attn = GlobalPatchAttention(d_model, d_t, patch_size)
        self.proj = nn.Linear(d_t, d_model)

    def forward(self, patch_sequence):
        B, N_p, P, D = patch_sequence.shape

        local_out = self.local_attn(patch_sequence)

        flattened = patch_sequence.reshape(B, N_p, P * D)
        global_out = self.global_attn(flattened)

        fused = local_out + global_out
        fused = self.proj(fused)

        return fused


class MAPTA(nn.Module):
    def __init__(self, d_model=64, d_t=32, k_route=3, patch_sizes=[2, 4, 6, 8],
                 kernel_sizes=[3, 5, 7], k_freq=1):
        super().__init__()
        self.d_model = d_model
        self.d_t = d_t
        self.k_route = k_route
        self.patch_sizes = patch_sizes
        self.total_patch_sizes = len(patch_sizes)

        self.seasonal_extractor = DFTSeasonalExtractor(d_model, k_freq)
        self.trend_extractor = MultiScaleTrendExtractor(d_model, kernel_sizes)
        self.router = NoisyGatingRouter(d_model, k_route, self.total_patch_sizes)

        self.dual_attentions = nn.ModuleDict({
            str(p): DualPatchAttention(d_model, d_t, p)
            for p in self.patch_sizes
        })

        self.transposed_convs = nn.ModuleDict({
            str(p): nn.ConvTranspose1d(d_model, d_model, kernel_size=p, stride=p)
            for p in patch_sizes
        })

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, z):
        B, T, N, D = z.shape
        device = z.device

        z = z.permute(0, 2, 1, 3).reshape(B * N, T, D)

        z_sea = self.seasonal_extractor(z)
        z_rem = z - z_sea
        z_trend = self.trend_extractor(z_rem)

        patch_candidates = torch.tensor(self.patch_sizes, device=device, dtype=torch.long)
        sparse_routing, selected_patches, routing_indices = self.router(z, z_sea, z_trend, patch_candidates)

        outputs = []
        for i, patch_size in enumerate(self.patch_sizes):
            weight_per_timestep = sparse_routing[:, :, i]
            weight = weight_per_timestep.unsqueeze(1)

            if not torch.any(weight > 0):
                continue

            patch_sequence = self._create_patches(z, patch_size)
            attn_out = self.dual_attentions[str(patch_size)](patch_sequence)
            attn_out = attn_out.permute(0, 2, 1)
            upsampled = self.transposed_convs[str(patch_size)](attn_out)

            if upsampled.shape[2] < T:
                upsampled = F.pad(upsampled, (0, T - upsampled.shape[2]))
            upsampled = upsampled[:, :, :T]

            outputs.append(upsampled * weight)

        if len(outputs) > 0:
            temporal_out = torch.stack(outputs, dim=0).sum(dim=0)
        else:
            temporal_out = torch.zeros_like(z.permute(0, 2, 1))

        temporal_out = temporal_out.reshape(B, N, T, D).permute(0, 2, 1, 3)

        return temporal_out

    def _create_patches(self, z, patch_size):
        B, T, D = z.shape

        padding = 0
        if T % patch_size != 0:
            padding = patch_size - (T % patch_size)
            z = F.pad(z.permute(0, 2, 1), (0, padding)).permute(0, 2, 1)

        T_padded = T + padding
        N_p = T_padded // patch_size

        patches = z.reshape(B, N_p, patch_size, D)

        return patches


class TraditionalTemporalAttention(nn.Module):
    def __init__(self, dim, dim_out, t_num_heads=2, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)

        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x
