"""
Training script for multimodal sarcasm detection
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

from utils import (
    setup_logging, get_device, seed_everything, AverageMeter, 
    count_parameters, format_time, save_results
)
from data_preprocessing import DataPreprocessor
from feature_extraction import MultimodalFeatureExtractor, FeatureFusion
from model import build_model


logger = setup_logging(level="INFO")


class TrainerConfig:
    """Training configuration"""
    
    def __init__(self, config: Dict):
        self.batch_size = int(config['training']['batch_size'])
        self.learning_rate = float(config['training']['learning_rate'])
        self.num_epochs = int(config['training']['num_epochs'])
        self.optimizer_name = config['training']['optimizer']
        self.loss_function = config['training']['loss_function']
        self.early_stopping_patience = int(config['training']['early_stopping_patience'])
        self.scheduler_name = config['training']['scheduler']


class SarcasmDetectionTrainer:
    """Trainer for multimodal sarcasm detection model"""
    
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: MultimodalFeatureExtractor,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame],
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data if test_data is not None else pd.DataFrame()
        self.config = config
        self.device = device
        
        # Training configuration
        self.train_config = TrainerConfig(config)
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
        self.early_stopping_counter = 0
        self.track_test_each_epoch = bool(config.get('training', {}).get('track_test_each_epoch', True))
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        if self.train_config.optimizer_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=1e-4
            )
        elif self.train_config.optimizer_name.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=1e-4
            )
        elif self.train_config.optimizer_name.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config.optimizer_name}")
    
    def _build_scheduler(self) -> object:
        """Build learning rate scheduler"""
        if self.train_config.scheduler_name.lower() == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.num_epochs
            )
        elif self.train_config.scheduler_name.lower() == 'step':
            return StepLR(
                self.optimizer,
                step_size=5,
                gamma=0.1
            )
        else:
            return None
    
    def _extract_batch_features_from(
        self,
        data_df: pd.DataFrame,
        batch_indices: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features for a batch from the provided dataframe"""
        batch_data = []
        labels = []

        for idx in batch_indices:
            row = data_df.iloc[idx]
            
            # Determine video path
            video_type = row['video_type']
            video_base = row['video_base']
            
            if video_type == 'c':
                video_path = os.path.join(
                    self.config['data']['raw_data_path'],
                    'context_videos',
                    f"{video_base}.mp4"
                )
            else:
                video_path = os.path.join(
                    self.config['data']['raw_data_path'],
                    'utterance_videos',
                    f"{video_base}.mp4"
                )
            
            batch_data.append({
                'video_path': video_path,
                'text': row['SENTENCE'],
                'start_time': 0.0,
                'end_time': row.get('end_time_seconds', None)
            })
            
            labels.append(int(row['Sarcasm']))
        
        # Extract features
        video_features, text_features, audio_features = self.feature_extractor.extract_batch_features(batch_data)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        return video_features, text_features, audio_features, labels_tensor
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Accuracy')
        
        n_samples = len(self.train_data)
        n_batches = (n_samples + self.train_config.batch_size - 1) // self.train_config.batch_size
        indices = np.random.permutation(n_samples)

        pbar = tqdm(total=n_batches, desc="Training")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.train_config.batch_size
            end_idx = min(start_idx + self.train_config.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx].tolist()
            
            try:
                # Extract features
                video_features, text_features, audio_features, labels = self._extract_batch_features_from(
                    self.train_data,
                    batch_indices
                )
                
                # Forward pass
                outputs = self.model(video_features, text_features, audio_features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Metrics
                _, preds = outputs.max(1)
                acc = (preds == labels).float().mean()
                
                loss_meter.update(loss.item(), len(batch_indices))
                acc_meter.update(acc.item(), len(batch_indices))
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {str(e)}")
            
            pbar.update(1)
        
        pbar.close()
        
        return {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg
        }
    
    def _evaluate_split(self, data_df: pd.DataFrame, split_name: str = "Validation") -> Dict[str, float]:
        """Evaluate loss/accuracy on a given split."""
        self.model.eval()

        if data_df is None or len(data_df) == 0:
            return {
                'loss': 0.0,
                'accuracy': 0.0
            }
        
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Accuracy')
        
        n_samples = len(data_df)
        n_batches = (n_samples + self.train_config.batch_size - 1) // self.train_config.batch_size
        
        pbar = tqdm(total=n_batches, desc=f"{split_name}")
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.train_config.batch_size
                end_idx = min(start_idx + self.train_config.batch_size, n_samples)
                batch_indices = list(range(start_idx, end_idx))
                
                try:
                    # Extract features from the provided split
                    video_features, text_features, audio_features, labels = self._extract_batch_features_from(
                        data_df,
                        batch_indices
                    )
                    
                    # Forward pass
                    outputs = self.model(video_features, text_features, audio_features)
                    loss = self.criterion(outputs, labels)
                    
                    # Metrics
                    _, preds = outputs.max(1)
                    acc = (preds == labels).float().mean()
                    
                    loss_meter.update(loss.item(), len(batch_indices))
                    acc_meter.update(acc.item(), len(batch_indices))
                
                except Exception as e:
                    logger.warning(f"Error in {split_name.lower()} batch {batch_idx}: {str(e)}")
                
                pbar.update(1)
        
        pbar.close()
        
        return {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        return self._evaluate_split(self.val_data, split_name="Validating")

    def test_epoch(self) -> Dict[str, float]:
        """Evaluate on test set at epoch granularity."""
        return self._evaluate_split(self.test_data, split_name="Testing")

    def _save_training_artifacts(self, include_plots: bool = True):
        """Persist per-epoch metrics and plots for training/validation/testing."""
        results_dir = self.config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        history_json = os.path.join(results_dir, 'training_history.json')
        history_json_ts = os.path.join(results_dir, f'training_history_{run_timestamp}.json')
        with open(history_json, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)
        with open(history_json_ts, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)

        history_df = pd.DataFrame(self.train_history)
        history_csv = os.path.join(results_dir, 'training_history.csv')
        history_csv_ts = os.path.join(results_dir, f'training_history_{run_timestamp}.csv')
        epoch_report_txt = os.path.join(results_dir, 'per_epoch_metrics.txt')
        epoch_report_txt_ts = os.path.join(results_dir, f'per_epoch_metrics_{run_timestamp}.txt')
        history_df.to_csv(history_csv, index=False)
        history_df.to_csv(history_csv_ts, index=False)

        # Also save a human-readable table for quick manual inspection.
        history_table = history_df.to_string(index=False)
        with open(epoch_report_txt, 'w', encoding='utf-8') as f:
            f.write(history_table + "\n")
        with open(epoch_report_txt_ts, 'w', encoding='utf-8') as f:
            f.write(history_table + "\n")

        # Plot train/val/test accuracy per epoch
        epochs = self.train_history['epoch']
        if include_plots and epochs:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.train_history['train_acc'], label='Train Acc', marker='o')
            plt.plot(epochs, self.train_history['val_acc'], label='Val Acc', marker='o')
            if any(v is not None for v in self.train_history['test_acc']):
                plt.plot(epochs, self.train_history['test_acc'], label='Test Acc', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy per Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.savefig(os.path.join(results_dir, 'training_accuracy_curve.png'), dpi=150)
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.train_history['train_loss'], label='Train Loss', marker='o')
            plt.plot(epochs, self.train_history['val_loss'], label='Val Loss', marker='o')
            if any(v is not None for v in self.train_history['test_loss']):
                plt.plot(epochs, self.train_history['test_loss'], label='Test Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss per Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.savefig(os.path.join(results_dir, 'training_loss_curve.png'), dpi=150)
            plt.close(fig)

        logger.info(f"Per-epoch JSON (latest): {os.path.abspath(history_json)}")
        logger.info(f"Per-epoch JSON (timestamped): {os.path.abspath(history_json_ts)}")
        logger.info(f"Per-epoch CSV (latest): {os.path.abspath(history_csv)}")
        logger.info(f"Per-epoch CSV (timestamped): {os.path.abspath(history_csv_ts)}")
        logger.info(f"Per-epoch text report (latest): {os.path.abspath(epoch_report_txt)}")
        logger.info(f"Per-epoch text report (timestamped): {os.path.abspath(epoch_report_txt_ts)}")
        logger.info("Training curve plots saved to results directory")
    
    def train(self) -> Dict:
        """Train the model"""
        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        logger.info(f"Batch size: {self.train_config.batch_size}")
        logger.info(f"Learning rate: {self.train_config.learning_rate}")
        logger.info(f"Num epochs: {self.train_config.num_epochs}")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        for epoch in range(self.train_config.num_epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{self.train_config.num_epochs} ---")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Store history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])

            if self.track_test_each_epoch and self.test_data is not None and len(self.test_data) > 0:
                test_metrics = self.test_epoch()
                self.train_history['test_loss'].append(test_metrics['loss'])
                self.train_history['test_acc'].append(test_metrics['accuracy'])
                logger.info(
                    f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}"
                )
            else:
                self.train_history['test_loss'].append(None)
                self.train_history['test_acc'].append(None)

            # Persist running epoch history so results are available even if interrupted.
            self._save_training_artifacts(include_plots=False)
            
            # Learning rate scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.early_stopping_counter = 0
                self._save_checkpoint(epoch, val_metrics)
                logger.info(f"✓ Best model saved with accuracy: {self.best_val_acc:.4f}")
            else:
                self.early_stopping_counter += 1
                logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.train_config.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.train_config.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info("Training Completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Training time: {format_time(training_time)}")
        logger.info("=" * 70)

        self._save_training_artifacts(include_plots=True)
        
        return {
            'best_val_acc': self.best_val_acc,
            'training_time': training_time,
            'history': self.train_history,
            'best_model_path': self.best_model_path
        }
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_dir = self.config['paths']['models_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(
            checkpoint_dir,
            f"best_model_{timestamp}_epoch{epoch}.pth"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, model_file)
        
        self.best_model_path = model_file

