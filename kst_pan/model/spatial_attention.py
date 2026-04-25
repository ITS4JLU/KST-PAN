import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging


BIG_CONSTANT = 1e8


def create_projection_matrix(m, d, seed=0, scaling=0):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        unstructured_block = torch.randn((d, d))
        q, _ = torch.qr(unstructured_block)
        q = q.t()
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        unstructured_block = torch.randn((d, d))
        q, _ = torch.qr(unstructured_block)
        q = q.t()
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}")

    return torch.matmul(torch.diag(multiplier), final_matrix)


def random_fourier_features(query, key, value, projection_matrix, d_s_prime, tau=0.25):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)

    logging.debug(f"rff: query.shape={query.shape}, key.shape={key.shape}, value.shape={value.shape}")
    logging.debug(f"rff: projection_matrix.shape={projection_matrix.shape}")

    query_prime = math.sqrt(2.0 / d_s_prime) * torch.cos(
        torch.matmul(query, projection_matrix.t())
    )
    key_prime = math.sqrt(2.0 / d_s_prime) * torch.cos(
        torch.matmul(key, projection_matrix.t())
    )

    logging.debug(f"rff: after cos query_prime.shape={query_prime.shape}, key_prime.shape={key_prime.shape}")

    query_prime = query_prime.permute(1, 0, 2, 3)
    key_prime = key_prime.permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)

    logging.debug(f"rff: after permute query_prime.shape={query_prime.shape}, key_prime.shape={key_prime.shape}")

    numerator_sum = torch.einsum("nbhm,nbhd->bhmd", key_prime, value)
    logging.debug(f"rff: numerator_sum.shape={numerator_sum.shape}")
    numerator = torch.einsum("nbhm,bhmd->nbhd", query_prime, numerator_sum)
    logging.debug(f"rff: numerator.shape={numerator.shape}")

    all_ones = torch.ones([key_prime.shape[0]]).to(query.device)
    denominator_sum = torch.einsum("nbhm,n->bhm", key_prime, all_ones)
    denominator = torch.einsum("nbhm,bhm->nbh", query_prime, denominator_sum)

    numerator = numerator.permute(1, 0, 2, 3)
    denominator = denominator.permute(1, 0, 2)
    denominator = torch.unsqueeze(denominator, len(denominator.shape))
    z_output = numerator / (denominator + 1e-8)

    logging.debug(f"rff: final z_output.shape={z_output.shape}")

    return z_output


class DTWMaskCalculator(nn.Module):
    def __init__(self, dtw_delta=5, sigma_dist=1.0, device=torch.device('cpu')):
        super().__init__()
        self.dtw_delta = dtw_delta
        self.sigma_dist = sigma_dist
        self.device = device

    def compute_dtw_mask(self, dtw_matrix):
        Theta = torch.from_numpy(dtw_matrix).float().to(self.device)
        mask = torch.exp(-torch.square(Theta / self.sigma_dist))
        return mask

    def forward(self, adj_mx, sd_mx, dtw_matrix, cluster_labels=None, epsilon_clu=0.1):
        num_nodes = adj_mx.shape[0]
        if isinstance(adj_mx, np.ndarray):
            adj_mx = torch.from_numpy(adj_mx).float().to(self.device)
        if isinstance(sd_mx, np.ndarray):
            sd_mx = torch.from_numpy(sd_mx).float().to(self.device)

        A_phys = torch.where(
            sd_mx < self.dtw_delta,
            torch.exp(-torch.square(sd_mx / self.sigma_dist)),
            torch.zeros_like(sd_mx, device=self.device)
        )

        if cluster_labels is not None:
            cluster_labels = torch.from_numpy(cluster_labels).long().to(self.device)
            A_ext = torch.zeros_like(adj_mx, device=self.device)
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if cluster_labels[u] == cluster_labels[v]:
                        A_ext[u, v] = 1.0
                    else:
                        A_ext[u, v] = epsilon_clu
        else:
            A_ext = torch.ones_like(adj_mx, device=self.device)

        dtw_mask = self.compute_dtw_mask(dtw_matrix)

        M_theta = A_phys * A_ext * dtw_mask

        return M_theta


class DGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads,
                 nb_random_features=16, use_bias=True, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.nb_random_features = nb_random_features

        self.Wq = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.Wk = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.Wv = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.Wo = nn.Linear(out_channels, out_channels, bias=use_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, z, edge_index, M_theta, projection_matrix, tau=0.25):
        B, N, T, D = z.shape

        query = self.Wq(z).reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3, 4)
        key = self.Wk(z).reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3, 4)
        value = self.Wv(z).reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3, 4)

        if projection_matrix is None:
            z_output = self._standard_attention(query, key, value, M_theta)
        else:
            z_output = self._kernelized_attention(query, key, value, M_theta, projection_matrix, tau)

        logging.debug(f"DGATBlock: z_output after attention: {z_output.shape}")
        z_output = z_output.permute(0, 2, 1, 3, 4).reshape(B, N, T, -1)
        logging.debug(f"DGATBlock: z_output after reshape: {z_output.shape}")
        z_output = self.Wo(z_output)
        logging.debug(f"DGATBlock: z_output after Wo: {z_output.shape}")

        return z_output

    def _standard_attention(self, query, key, value, M_theta):
        scale = self.head_dim ** -0.5
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale

        M_theta = M_theta.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        attn = attn + torch.log(M_theta + 1e-8)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, value)
        return output

    def _kernelized_attention(self, query, key, value, M_theta, projection_matrix, tau):
        B, T, N, H, D_head = query.shape
        d_s_prime = projection_matrix.shape[0]

        query_prime = query.permute(2, 0, 3, 4, 1).reshape(N, B * T, H, D_head)
        key_prime = key.permute(2, 0, 3, 4, 1).reshape(N, B * T, H, D_head)
        value_prime = value.permute(2, 0, 3, 4, 1).reshape(N, B * T, H, D_head)

        z_output = random_fourier_features(
            query_prime, key_prime, value_prime,
            projection_matrix, d_s_prime, tau
        )

        logging.debug(f"_kernelized_attention:")
        logging.debug(f"  z_output.shape after rff: {z_output.shape}")
        logging.debug(f"  B={B}, T={T}, N={N}, H={H}, D_head={D_head}")

        z_output = z_output.reshape(N, B, T, H, D_head).permute(1, 2, 0, 3, 4)

        logging.debug(f"  z_output.shape after reshape/permute: {z_output.shape}")
        logging.debug(f"  M_theta.shape: {M_theta.shape}")

        M_theta_expanded = M_theta.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        logging.debug(f"  M_theta_expanded.shape: {M_theta_expanded.shape}")

        if z_output.shape[2] == M_theta_expanded.shape[2] and z_output.shape[3] == M_theta_expanded.shape[3]:
            z_output = z_output * M_theta_expanded
            logging.debug(f"  Applied M_theta multiplication")
        else:
            logging.debug(f"  Skipped M_theta multiplication due to shape mismatch")

        return z_output


class PKSAConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads,
                 nb_random_features=16, use_gumbel=False, nb_gumbel_sample=10,
                 rb_order=0, rb_trans='sigmoid', use_edge_loss=False,
                 dtw_delta=5, num_dgat_rounds=2):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss
        self.num_dgat_rounds = num_dgat_rounds

        self.dgat_blocks = nn.ModuleList([
            DGATBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                nb_random_features=nb_random_features
            )
            for i in range(num_dgat_rounds)
        ])

        if rb_order >= 1:
            self.b = nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)
            nn.init.constant_(self.b, 0.1)

    def reset_parameters(self):
        for block in self.dgat_blocks:
            block.Wq.reset_parameters()
            block.Wk.reset_parameters()
            block.Wv.reset_parameters()
            block.Wo.reset_parameters()
        if self.rb_order >= 1:
            if self.rb_trans == 'sigmoid':
                nn.init.constant_(self.b, 0.1)
            elif self.rb_trans == 'identity':
                nn.init.constant_(self.b, 1.0)

    def forward(self, z, edge_index, M_theta, tau=1.0, mask=None):
        B, N = z.size(0), z.size(1)
        B, N, T, D = z.shape

        for r in range(self.num_dgat_rounds):
            head_dim = self.out_channels // self.num_heads
            seed = torch.ceil(torch.abs(torch.sum(z) * BIG_CONSTANT)).to(torch.int32)
            projection_matrix = create_projection_matrix(
                self.nb_random_features, head_dim, seed=seed).to(z.device)

            z = self.dgat_blocks[r](z, edge_index, M_theta, projection_matrix, tau)

            if self.rb_order >= 1 and r == self.num_dgat_rounds - 1:
                z = z + self._compute_relational_bias(z, edge_index)

        if mask is not None:
            # mask: [N, N] -> 简化为节点掩码 [N]
            # 使用对角线元素作为节点掩码
            node_mask = torch.diag(mask).unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # [1, N, 1, 1]
            z = z * node_mask  # [B, N, T, D] * [1, N, 1, 1] -> [B, N, T, D]

        return z

        return z

    def _compute_relational_bias(self, z, edge_index):
        B, T, N, D = z.shape

        row, col = edge_index
        z_reshaped = z.permute(0, 2, 1, 3).reshape(B * T, N, D)

        d_in = self._compute_degree(col, N).to(z.device)
        d_norm_in = (1. / d_in[col]).sqrt()
        d_out = self._compute_degree(row, N).to(z.device)
        d_norm_out = (1. / d_out[row]).sqrt()

        bias = torch.sigmoid(self.b[0]).mean()

        relational_bias = torch.zeros(B, T, N, N).to(z.device)
        relational_bias[:, :, row, col] = bias * d_norm_in * d_norm_out

        output = torch.matmul(relational_bias, z_reshaped)
        output = output.reshape(B, T, N, D)

        return output

    def _compute_degree(self, indices, num_nodes):
        degree = torch.zeros(num_nodes)
        degree.scatter_add_(0, indices, torch.ones_like(indices).float())
        return degree
