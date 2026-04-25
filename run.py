import os
import torch
import argparse
import numpy as np
import random
import json

from kst_pan.model import KST_PAN
from kst_pan.data import KST_PANDataset
from kst_pan.train import KST_PANTrainer, KST_PANEvaluator
from kst_pan.model.utils import calculate_laplacian_pe
import logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def merge_config(args_config):
    # Base config path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = args_config.get('dataset', 'PeMS08')
    json_path = os.path.join(base_dir, 'kst_pan', 'config', 'dataset_config', f'{dataset}.json')
    
    # Load dataset-specific config if it exists
    if os.path.exists(json_path):
        logging.info(f"Loading dataset config from: {json_path}")
        with open(json_path, 'r') as f:
            dataset_config = json.load(f)
            
        # Merge dicts: Command line arguments override JSON configs
        # Filter out default values that weren't explicitly set (optional, but let's keep it simple)
        for k, v in dataset_config.items():
            if k not in args_config or args_config[k] is None:
                args_config[k] = v
            # If the user passed default via argparse, we might want JSON to take precedence
            # But since argparse fills defaults, let's let JSON overwrite specific keys
            elif k in ['batch_size', 'input_window', 'output_window', 'max_epoch', 'seed', 'embed_dim']:
                args_config[k] = v
                
    else:
        logging.warning(f"Warning: No config file found for dataset '{dataset}' at {json_path}")
        
    return args_config


def train_model(config):
    config = merge_config(config)
    set_seed(config.get('seed', 0))

    # Determine device based on config and availability
    requested_device = config.get('device', 'cuda')
    if 'cuda' in requested_device and torch.cuda.is_available():
        try:
            device = torch.device(requested_device)
        except RuntimeError:
            logging.warning(f"Warning: Requested CUDA device {requested_device} not available. Using default CUDA device.")
            device = torch.device('cuda') # Fallback to default CUDA
    else:
        if 'cuda' in requested_device and not torch.cuda.is_available():
            logging.warning("Warning: CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
    config['device'] = device
    logging.info(f"Using device: {device}")

    dataset = config.get('dataset', 'PeMS08')
    cache_dir = config.get('cache_dir', './cache')
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d%H')
    config['cache_dir'] = os.path.join(cache_dir, dataset)
    config['exp_id'] = timestamp

    dataset = KST_PANDataset(config)
    train_loader, val_loader, test_loader = dataset.get_data()
    data_feature = dataset.get_data_feature()

    if 'adj_mx' not in data_feature or data_feature['adj_mx'] is None:
        raise ValueError("Adjacency matrix is required for KST_PAN")

    adj_mx = data_feature['adj_mx']
    lape_dim = config.get('lape_dim', 8)
    lap_mx = calculate_laplacian_pe(adj_mx, lape_dim).to(device)

    model = KST_PAN(config, data_feature)
    model.lap_mx = lap_mx

    trainer = KST_PANTrainer(model, config)
    trainer.train(train_loader, val_loader)

    evaluator = KST_PANEvaluator(model, config)
    test_metrics = evaluator.evaluate(test_loader, scaler=data_feature['scaler'])
    logging.info(f"Test Metrics: {test_metrics}")

    return model, test_metrics, trainer


def main():
    parser = argparse.ArgumentParser(description='KST_PAN Traffic Prediction')

    parser.add_argument('--dataset', type=str, default='PeMS07',
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Model embedding dimension')
    parser.add_argument('--skip_dim', type=int, default=256,
                        help='Skip connection dimension')
    parser.add_argument('--enc_depth', type=int, default=6,
                        help='Number of encoder layers')

    parser.add_argument('--input_window', type=int, default=12,
                        help='Input time window')
    parser.add_argument('--output_window', type=int, default=12,
                        help='Output time window')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max_epoch', type=int, default=300,
                        help='Maximum training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')

    parser.add_argument('--s_attn_size', type=int, default=3,
                        help='Spatial attention size')
    parser.add_argument('--n_cluster', type=int, default=8,
                        help='Number of clusters for KShape')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='Output dimension')
    parser.add_argument('--sem_attn_mode', type=str, default='dynamic',
                        help='Semantic attention mode: dynamic or traditional')
    parser.add_argument('--t_attn_mode', type=str, default='patch',
                        help='Temporal attention mode: patch or traditional')
    
    parser.add_argument('--train_rate', type=float, default=0.7,
                        help='Train data rate')
    parser.add_argument('--eval_rate', type=float, default=0.1,
                        help='Eval data rate')

    args = parser.parse_args()
    config = vars(args)

    try:
        model, metrics, trainer = train_model(config)

        trainer.logger.info("\n=== Final Results ===")
        trainer.logger.info(f"MAE: {metrics['MAE']:.4f}")
        trainer.logger.info(f"RMSE: {metrics['RMSE']:.4f}")
        trainer.logger.info(f"MAPE: {metrics['MAPE']:.4f}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if 'trainer' in locals() and trainer is not None:
                trainer.logger.info("CUDA cache emptied.")
            else:
                logging.info("CUDA cache emptied.")


if __name__ == '__main__':
    main()
