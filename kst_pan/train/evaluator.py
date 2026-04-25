import os
import time
import numpy as np
import torch


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def evaluate_predictions(y_true, y_pred, null_val=0.0):
    mask = (y_true != null_val)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    mae = np.mean(np.abs(y_true_masked - y_pred_masked))
    rmse = np.sqrt(np.mean(np.square(y_true_masked - y_pred_masked)))
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / (y_true_masked + 1e-5))) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


class KST_PANEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.get('device', torch.device('cpu'))
        self.output_dim = config.get('output_dim', 1)

        self.exp_id = config.get('exp_id', 'default')
        self.cache_dir = config.get('cache_dir', './cache')
        self.evaluate_res_dir = os.path.join(self.cache_dir, self.exp_id, 'evaluate_cache')
        ensure_dir(self.evaluate_res_dir)

    def evaluate(self, dataloader, scaler=None):
        self.model.eval()
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                batch = {'X': x}
                lap_mx = getattr(self.model, 'lap_mx', None)

                y_pred = self.model(batch, lap_mx)
                y_pred = y_pred.cpu().numpy()
                y = y.numpy()

                y_true_all.append(y)
                y_pred_all.append(y_pred)

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)

        if scaler is not None:
            y_true_all = scaler.inverse_transform(y_true_all[..., :self.output_dim])
            y_pred_all = scaler.inverse_transform(y_pred_all[..., :self.output_dim])
        else:
            y_true_all = y_true_all[..., :self.output_dim]
            y_pred_all = y_pred_all[..., :self.output_dim]

        metrics = evaluate_predictions(y_true_all, y_pred_all)

        return metrics

    def save_pred(self, y_true, y_pred, filename):
        ensure_dir(self.evaluate_res_dir)
        pred_path = os.path.join(self.evaluate_res_dir, f'{filename}_predictions.npz')
        np.savez(pred_path, y_true=y_true, y_pred=y_pred)
        print(f"Predictions saved to {pred_path}")

    def save_metrics(self, metrics, filename):
        ensure_dir(self.evaluate_res_dir)
        metrics_path = os.path.join(self.evaluate_res_dir, f'{filename}_average.csv')
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key},{value}\n")
        print(f"Metrics saved to {metrics_path}")
