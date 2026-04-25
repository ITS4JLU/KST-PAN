import os
import pandas as pd
import numpy as np
import torch
from fastdtw import fastdtw
from tqdm import tqdm
import logging
# from tslearn.clustering import KShape # Replaced with GPU K-Means

def _pytorch_kmeans(data, n_cluster, n_iter=20, device=None):
    """
    GPU-accelerated K-Means implementation using PyTorch.
    Approximates KShape for performance.
    """
    data_tensor = torch.from_numpy(data).float()
    if device is not None and 'cuda' in str(device):
        data_tensor = data_tensor.to(device)
    n_samples, n_features = data_tensor.shape

    initial_indices = torch.randperm(n_samples, device=device)[:n_cluster]
    centers = data_tensor[initial_indices]

    for _ in range(n_iter):
        # E-step: Assign points to the nearest cluster
        distances = torch.cdist(data_tensor, centers)
        cluster_assignments = torch.argmin(distances, dim=1)

        # M-step: Update cluster centers (vectorized)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(cluster_assignments, minlength=n_cluster).float().unsqueeze(1)
        
        new_centers.scatter_add_(0, cluster_assignments.unsqueeze(1).expand(-1, n_features), data_tensor)
        
        # Avoid division by zero for empty clusters
        new_centers /= (counts + 1e-8)

        # Handle empty clusters by re-initializing them to random data points
        empty_clusters_mask = (counts.squeeze() == 0)
        num_empty = empty_clusters_mask.sum().item()
        if num_empty > 0:
            new_centers[empty_clusters_mask] = data_tensor[torch.randperm(n_samples, device=device)[:num_empty]]

        if torch.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return centers.cpu().numpy()
from torch.utils.data import DataLoader


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data * self.std + self.mean


class MinMaxScaler:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class KST_PANDataset:
    def __init__(self, config):
        self.dataset = config.get('dataset', 'PeMS08')
        self.data_path = config.get('data_path', './raw_data')
        self.cache_dir = config.get('cache_dir', './cache')
        self.time_intervals = config.get('time_intervals', 300)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.output_dim = config.get('output_dim', 1)
        self.batch_size = config.get('batch_size', 16)
        self.num_workers = config.get('num_workers', 0)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.n_cluster = config.get("n_cluster", 16)
        self.device = config.get('device', 'cpu')
        self.cluster_max_iter = config.get("cluster_max_iter", 5)
        self.scaler_type = config.get('scaler_type', 'standard')
        self.cache_dataset = config.get('cache_dataset', True)

        self.adj_mx = None
        self.sd_mx = None
        self.sh_mx = None
        self.data = None
        self.num_nodes = 0
        self.feature_dim = 0
        self.ext_dim = 0
        self.scaler = None
        self.dtw_matrix = None
        self.pattern_keys = None

        self.train_rate = config.get('train_rate', 0.7)
        self.eval_rate = config.get('eval_rate', 0.1)

        self.dataset_cache_file = os.path.join(
            self.cache_dir, 'dataset_cache',
            f'point_based_{self.dataset}_{self.input_window}_{self.output_window}.npz'
        )
        ensure_dir(os.path.dirname(self.dataset_cache_file))

    def _load_geo(self):
        geo_file = os.path.join(self.data_path, self.dataset, f"{self.dataset}.geo")
        df = pd.read_csv(geo_file)
        self.num_nodes = len(df)
        self.geo_ids = df['geo_id'].values.tolist()
        self.geo_to_ind = {geo_id: ind for ind, geo_id in enumerate(self.geo_ids)}

    def _load_rel(self):
        rel_file = os.path.join(self.data_path, self.dataset, f"{self.dataset}.rel")
        df = pd.read_csv(rel_file)

        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes))
        self.sd_mx = np.zeros((self.num_nodes, self.num_nodes))

        for _, row in df.iterrows():
            if row['origin_id'] in self.geo_to_ind and row['destination_id'] in self.geo_to_ind:
                i = self.geo_to_ind[row['origin_id']]
                j = self.geo_to_ind[row['destination_id']]
                self.adj_mx[i, j] = 1.0
                self.sd_mx[i, j] = row.get('cost', 1.0)

        self.sh_mx = self.sd_mx.copy()
        self.sh_mx[self.sh_mx == 0] = np.inf
        np.fill_diagonal(self.sh_mx, 0)
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    self.sh_mx[i, j] = min(self.sh_mx[i, j], self.sh_mx[i, k] + self.sh_mx[k, j])

    def _load_dyna(self):
        dyna_file = os.path.join(self.data_path, self.dataset, f"{self.dataset}.dyna")
        df = pd.read_csv(dyna_file)

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by=['time', 'entity_id'])

        times = df['time'].unique()
        num_times = len(times)

        exclude_cols = ['dyna_id', 'type', 'time', 'entity_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_dim = len(feature_cols)

        data_3d = df[feature_cols].values.reshape(num_times, self.num_nodes, self.feature_dim)

        times_pd = pd.to_datetime(times)
        time_of_day = (times_pd.hour * 60 + times_pd.minute) / 1440.0
        day_of_week = times_pd.dayofweek / 6.0

        tod_feature = np.tile(time_of_day[:, np.newaxis, np.newaxis], (1, self.num_nodes, 1))
        dow_feature = np.tile(day_of_week[:, np.newaxis, np.newaxis], (1, self.num_nodes, 1))

        self.data = np.concatenate([data_3d, tod_feature, dow_feature], axis=-1)
        self.feature_dim += 2
        self.ext_dim = 2

    def _get_dtw(self):
        cache_path = os.path.join(self.cache_dir, 'dataset_cache', f'dtw_{self.dataset}.npy')
        ensure_dir(os.path.dirname(cache_path))

        if not os.path.exists(cache_path):
            logging.info("Calculating DTW matrix...")
            df = self.data[..., :self.output_dim]
            points_per_hour = 3600 // self.time_intervals
            data_mean = np.mean([
                df[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
                for i in range(df.shape[0] // (24 * points_per_hour))
            ], axis=0)

            dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
            for i in tqdm(range(self.num_nodes)):
                for j in range(i, self.num_nodes):
                    dtw_distance[i][j], _ = fastdtw(
                        data_mean[:, i, :], data_mean[:, j, :], radius=6
                    )
            for i in range(self.num_nodes):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(cache_path, dtw_distance)

        self.dtw_matrix = np.load(cache_path)
        logging.info(f"Loaded DTW matrix from {cache_path}")

    def _get_pattern_keys(self):
        # Use a different cache file for GPU K-Means results
        cache_path = os.path.join(self.cache_dir, 'dataset_cache',
                                  f'pattern_keys_{self.dataset}_{self.s_attn_size}_{self.n_cluster}_kmeans_gpu.npy')
        ensure_dir(os.path.dirname(cache_path))
        if not os.path.exists(cache_path):
            logging.info("Calculating Pattern Keys using GPU-accelerated K-Means...")
            x_train = self.data[:int(self.data.shape[0] * self.train_rate), ..., :self.output_dim]

            pattern_keys = []
            for i in range(self.output_dim):
                data = x_train[..., i]
                # Vectorized patch extraction using stride tricks for high performance
                shape = (data.shape[0] - self.s_attn_size + 1, self.s_attn_size) + data.shape[1:]
                strides = (data.strides[0],) + data.strides
                patches = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

                # Reshape for clustering: (num_samples * num_nodes, seq_len)
                patches = patches.transpose(0, 2, 1).reshape(-1, self.s_attn_size)

                # Subsample if the dataset is too large, increased limit for GPU
                if patches.shape[0] > 200000:
                    indices = np.random.choice(patches.shape[0], 200000, replace=False)
                    patches = patches[indices]

                # Call the GPU K-Means function
                cluster_centers = _pytorch_kmeans(
                    patches,
                    n_cluster=self.n_cluster,
                    device=self.device
                )
                pattern_keys.append(cluster_centers)

            self.pattern_keys = np.stack(pattern_keys, axis=-1)
            np.save(cache_path, self.pattern_keys)
        else:
            self.pattern_keys = np.load(cache_path)
        logging.info(f"Loaded Pattern Keys from {cache_path}")

    def _get_scalar(self, scaler_type, X_train):
        if scaler_type == 'standard':
            mean = np.mean(X_train, axis=(0, 1, 2))
            std = np.std(X_train, axis=(0, 1, 2))
            return StandardScaler(mean, std)
        elif scaler_type == 'minmax':
            min_val = np.min(X_train, axis=(0, 1, 2))
            max_val = np.max(X_train, axis=(0, 1, 2))
            return MinMaxScaler(min_val, max_val)
        else:
            return None

    def _generate_samples(self, data):
        n_samples = len(data) - self.input_window - self.output_window + 1
        if n_samples <= 0:
            return np.array([]), np.array([])

        x_shape = (n_samples, self.input_window) + data.shape[1:]
        y_shape = (n_samples, self.output_window) + data.shape[1:]

        X = np.empty(x_shape, dtype=data.dtype)
        Y = np.empty(y_shape, dtype=data.dtype)

        for i in range(n_samples):
            X[i] = data[i:i + self.input_window]
            Y[i] = data[i + self.input_window:i + self.input_window + self.output_window]

        return X, Y

    def _load_cache_train_val_test(self):
        print(f"Loading dataset from cache: {self.dataset_cache_file}")
        cat_data = np.load(self.dataset_cache_file)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        scaler_mean = cat_data['scaler_mean']
        scaler_std = cat_data['scaler_std']
        self.data = cat_data['data'] # Load data from cache
        self.adj_mx = cat_data['adj_mx'] # Load adj_mx from cache
        self.dtw_matrix = cat_data['dtw_matrix'] # Load dtw_matrix from cache
        self.pattern_keys = cat_data['pattern_keys'] # Load pattern_keys from cache
        self.num_nodes = cat_data['num_nodes'] # Load num_nodes from cache
        self.feature_dim = cat_data['feature_dim'] # Load feature_dim from cache
        self.ext_dim = cat_data['ext_dim'] # Load ext_dim from cache
        self.sd_mx = cat_data['sd_mx'] # Load sd_mx from cache
        self.sh_mx = cat_data['sh_mx'] # Load sh_mx from cache
        self.scaler = StandardScaler(scaler_mean, scaler_std)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _save_cache_train_val_test(self, x_train, y_train, x_val, y_val, x_test, y_test):
        print(f"Saving dataset to cache: {self.dataset_cache_file}")
        np.savez(self.dataset_cache_file,
                 x_train=x_train, y_train=y_train,
                 x_val=x_val, y_val=y_val,
                 x_test=x_test, y_test=y_test,
                 scaler_mean=self.scaler.mean,
                 scaler_std=self.scaler.std,
                 data=self.data,
                 adj_mx=self.adj_mx, # Add adj_mx to cache
                 dtw_matrix=self.dtw_matrix,
                 pattern_keys=self.pattern_keys,
                 num_nodes=self.num_nodes,
                 feature_dim=self.feature_dim,
                 ext_dim=self.ext_dim,
                 sd_mx=self.sd_mx, # Add sd_mx to cache
                 sh_mx=self.sh_mx) # Add sh_mx to cache
        print(f"Dataset cache saved at {self.dataset_cache_file}")

    def get_data(self):
        if self.cache_dataset and os.path.exists(self.dataset_cache_file):
            try:
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            except Exception as e:
                print(f"Cache load failed: {e}, regenerating data...")
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_data()
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = self._generate_data()

        train_data = list(zip(torch.FloatTensor(x_train), torch.FloatTensor(y_train)))
        eval_data = list(zip(torch.FloatTensor(x_val), torch.FloatTensor(y_val)))
        test_data = list(zip(torch.FloatTensor(x_test), torch.FloatTensor(y_test)))

        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.eval_dataloader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def _generate_data(self):
        if self.data is None:
            self._load_geo()
            self._load_rel()
            self._load_dyna()
            self._get_dtw()
            self._get_pattern_keys()

        total_time = self.data.shape[0]
        train_end = int(total_time * self.train_rate)
        val_end = int(total_time * (self.train_rate + self.eval_rate))

        train_data_raw = self.data[:train_end]
        val_data_raw = self.data[train_end:val_end]
        test_data_raw = self.data[val_end:]

        x_train, y_train = self._generate_samples(train_data_raw)
        x_val, y_val = self._generate_samples(val_data_raw)
        x_test, y_test = self._generate_samples(test_data_raw)

        self.scaler = self._get_scalar(self.scaler_type, x_train[..., :self.output_dim])

        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])

        if self.cache_dataset:
            self._save_cache_train_val_test(x_train, y_train, x_val, y_val, x_test, y_test)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_data_feature(self):
        return {
            "scaler": self.scaler,
            "adj_mx": self.adj_mx,
            "sd_mx": self.sd_mx,
            "sh_mx": self.sh_mx,
            "num_nodes": self.num_nodes,
            "feature_dim": self.feature_dim,
            "ext_dim": self.ext_dim,
            "output_dim": self.output_dim,
            "num_batches": len(self.train_dataloader) if self.train_dataloader else 0,
            "dtw_matrix": self.dtw_matrix,
            "pattern_keys": self.pattern_keys
        }
