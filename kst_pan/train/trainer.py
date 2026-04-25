import os
import time
import numpy as np
import torch
import torch.nn as nn
import logging


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class KST_PANTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.get('device', torch.device('cpu'))
        self.scaler = config.get('scaler')

        self.task_level = 0

        self.exp_id = config.get('exp_id', 'default')
        self.cache_dir = config.get('cache_dir', './cache')
        self.model_cache_dir = os.path.join(self.cache_dir, self.exp_id, 'model_cache')
        self.evaluate_res_dir = os.path.join(self.cache_dir, self.exp_id, 'evaluate_cache')
        ensure_dir(self.model_cache_dir)
        ensure_dir(self.evaluate_res_dir)

        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        log_file = os.path.join(self.evaluate_res_dir, f'training_{self.exp_id}.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(f"Using device: {self.device}")

        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 0.05)
        self.epochs = config.get('max_epoch', 300)
        self.patience = config.get('patience', 50)

        self.learner = config.get('learner', 'adam')
        self.lr_scheduler_type = config.get('lr_scheduler', 'cosine')
        self.lr_decay_ratio = config.get('lr_decay_ratio', 0.1)
        self.lr_warmup_epoch = config.get('lr_warmup_epoch', 5)
        self.lr_warmup_init = config.get('lr_warmup_init', 1e-6)
        self.lr_eta_min = config.get('lr_eta_min', 1e-4)

        self.clip_grad_norm = config.get('clip_grad_norm', True)
        self.max_grad_norm = config.get('max_grad_norm', 5)

        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.task_level = 0

        self.loss_function = config.get('loss_function', 'huber')
        self.optimizer_eps = config.get('optimizer_eps', 1e-8)

        self.model = self.model.to(self.device)
        self._build_optimizer()
        self._build_lr_scheduler()

    def _build_optimizer(self):
        if self.learner.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.learner.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.learner.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate,
                momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

    def _build_lr_scheduler(self):
        if self.lr_scheduler_type.lower() == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=self.lr_eta_min
            )
        elif self.lr_scheduler_type.lower() == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=self.lr_decay_ratio
            )
        elif self.lr_scheduler_type.lower() == 'multistep':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[100, 200], gamma=self.lr_decay_ratio
            )
        else:
            self.lr_scheduler = None

    def _get_loss_func(self, set_loss='masked_mae'):
        if set_loss.lower() == 'mae':
            return self._masked_mae_loss
        elif set_loss.lower() == 'mse':
            return self._masked_mse_loss
        elif set_loss.lower() == 'huber':
            return self._huber_loss
        elif set_loss.lower() == 'quantile':
            return self._quantile_loss
        else:
            return self._masked_mae_loss

    def _masked_mae_loss(self, y_true, y_pred, null_val=0.0):
        mask = (y_true != null_val).float()
        mask = mask.expand_as(y_true)
        loss = torch.abs(y_true - y_pred) * mask
        return loss.mean()

    def _masked_mse_loss(self, y_true, y_pred, null_val=0.0):
        mask = (y_true != null_val).float()
        mask = mask.expand_as(y_true)
        loss = torch.square(y_true - y_pred) * mask
        return loss.mean()

    def _huber_loss(self, y_true, y_pred, delta=1.0):
        residual = torch.abs(y_true - y_pred)
        condition = residual < delta
        quadratic = torch.where(condition, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
        return quadratic.mean()

    def _quantile_loss(self, y_true, y_pred, delta=0.25):
        residual = y_true - y_pred
        condition = residual >= 0
        return torch.where(condition, delta * residual, (delta - 1) * residual).mean()

    def save_model(self, cache_name):
        ensure_dir(self.model_cache_dir)
        model_path = os.path.join(self.model_cache_dir, f'{cache_name}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'task_level': self.task_level
        }, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, cache_name):
        model_path = os.path.join(self.model_cache_dir, f'{cache_name}.pth')
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            return False
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.task_level = checkpoint.get('task_level', 0)
        self.logger.info(f"Model loaded from {model_path}")
        return True

    def train(self, train_dataloader, eval_dataloader=None):
        self.logger.info(f"Start training for {self.epochs} epochs...")
        best_loss = float('inf')
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, (x, y) in enumerate(train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                batch = {'X': x, 'y': y}
                lap_mx = getattr(self.model, 'lap_mx', None)

                if self.use_curriculum_learning:
                    if (epoch * num_batches + batch_idx) % self.step_size == 0:
                        if self.task_level < self.model.output_window:
                            self.task_level += 1
                            self.logger.info(f"Epoch {epoch}: task_level increase to {self.task_level}")

                y_pred = self.model(batch, lap_mx)

                loss_func = self._get_loss_func('masked_mae')
                if self.use_curriculum_learning:
                    loss = loss_func(y_pred[:, :self.task_level, :, :],
                                   y[:, :self.task_level, :, :])
                else:
                    loss = loss_func(y_pred, y)

                self.optimizer.zero_grad()
                loss.backward()

                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                self.optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {avg_train_loss:.4f}")

            if eval_dataloader is not None:
                val_loss = self.evaluate(eval_dataloader)
                self.logger.info(f"Epoch {epoch}/{self.epochs} - Val Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    self.save_model(f'best_model')
                else:
                    wait += 1
                    if wait >= self.patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

            if epoch % 10 == 0:
                self.save_model(f'epoch_{epoch}')

        self.logger.info("Training completed!")
        return best_loss

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                batch = {'X': x, 'y': y}
                lap_mx = getattr(self.model, 'lap_mx', None)

                y_pred = self.model(batch, lap_mx)
                loss = self._masked_mae_loss(y_pred, y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def predict(self, dataloader):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                batch = {'X': x}
                lap_mx = getattr(self.model, 'lap_mx', None)
                y_pred = self.model(batch, lap_mx)
                predictions.append(y_pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)
