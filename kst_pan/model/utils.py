import numpy as np
import torch
import scipy.sparse as sp


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # Fix: clip zeros to 1 before inversion to avoid divide-by-zero warning
    d = np.clip(d, a_min=1.0, a_max=None)
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num


def calculate_laplacian_pe(adj_mx, lape_dim):
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = torch.from_numpy(
        EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    ).float()
    laplacian_pe.require_grad = False
    return laplacian_pe


def calculate_adjacency_distance(adj_mx, weight_adj_epsilon=0.0):
    sd_mx = adj_mx.copy()
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))
    adj_mx[adj_mx < weight_adj_epsilon] = 0
    return adj_mx, sd_mx


def calculate_shortest_path(adj_mx, num_nodes, type_short_path="hop"):
    sh_mx = adj_mx.copy()
    if type_short_path == "hop":
        sh_mx[sh_mx > 0] = 1
        sh_mx[sh_mx == 0] = 511
        for i in range(num_nodes):
            sh_mx[i, i] = 0
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)
    elif type_short_path == "dist":
        sh_mx[adj_mx == 0] = np.inf
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j])
    return sh_mx


def masked_mae_loss(y_true, y_pred, null_val=0.0):
    mask = (y_true != null_val).float()
    mask = mask.expand_as(y_true)
    return torch.abs(y_true - y_pred) * mask


def masked_mse_loss(y_true, y_pred, null_val=0.0):
    mask = (y_true != null_val).float()
    mask = mask.expand_as(y_true)
    return torch.square(y_true - y_pred) * mask


def huber_loss(y_true, y_pred, delta=1.0):
    residual = torch.abs(y_true - y_pred)
    condition = residual < delta
    quadratic = torch.where(condition, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
    return quadratic.mean()


def quantile_loss(y_true, y_pred, delta=0.25):
    residual = y_true - y_pred
    condition = residual >= 0
    return torch.where(condition, delta * residual, (delta - 1) * residual).mean()
