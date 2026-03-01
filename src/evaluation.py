"""
Evaluation and testing script for multimodal sarcasm detection
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, auc
)
import json
from datetime import datetime
from tqdm import tqdm

from utils import setup_logging, save_results

logger = setup_logging(level="INFO")


class SarcasmDetectionEvaluator:
    """Evaluator for multimodal sarcasm detection model"""
    
    def __init__(
        self,
        model: nn.Module,
        feature_extractor,
        test_data: pd.DataFrame,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.test_data = test_data
        self.config = config
        self.device = device
        
        self.model.eval()
        self.predictions = []
        self.probabilities = []
        self.ground_truth = []

    @staticmethod
    def _empty_metrics() -> Dict:
        """Return a safe empty metrics structure for zero-sample evaluation."""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'precision_per_class': [0.0, 0.0],
            'recall_per_class': [0.0, 0.0],
            'f1_per_class': [0.0, 0.0],
            'roc_auc': None,
            'confusion_matrix': [[0, 0], [0, 0]],
            'classification_report': {},
            'total_samples': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'sarcasm_distribution': {
                'not_sarcastic': 0,
                'sarcastic': 0
            }
        }
    
    def _extract_batch_features(self, batch_indices: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features for a batch"""
        batch_data = []
        labels = []
        
        for idx in batch_indices:
            row = self.test_data.iloc[idx]
            
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
    
    def evaluate(self) -> Dict:
        """Evaluate model on test set"""
        logger.info("=" * 70)
        logger.info("Starting Evaluation on Test Set")
        logger.info("=" * 70)
        
        batch_size = self.config['training']['batch_size']
        n_samples = len(self.test_data)

        if n_samples == 0:
            logger.warning("Test set is empty. Returning empty metrics.")
            self.predictions = np.array([], dtype=np.int64)
            self.probabilities = np.array([], dtype=np.float32).reshape(0, 2)
            self.ground_truth = np.array([], dtype=np.int64)
            metrics = self._empty_metrics()
            self._print_metrics(metrics)
            return metrics

        n_batches = (n_samples + batch_size - 1) // batch_size
        
        pbar = tqdm(total=n_batches, desc="Evaluating")
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = list(range(start_idx, end_idx))
                
                try:
                    # Extract features
                    video_features, text_features, audio_features, labels = self._extract_batch_features(batch_indices)
                    
                    # Forward pass
                    outputs = self.model(video_features, text_features, audio_features)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = outputs.max(1)
                    
                    # Store results
                    self.predictions.extend(preds.cpu().numpy())
                    self.probabilities.extend(probs.cpu().numpy())
                    self.ground_truth.extend(labels.cpu().numpy())
                
                except Exception as e:
                    logger.warning(f"Error in evaluation batch {batch_idx}: {str(e)}")
                
                pbar.update(1)
        
        pbar.close()
        
        # Convert to numpy
        self.predictions = np.array(self.predictions)
        self.probabilities = np.array(self.probabilities)
        self.ground_truth = np.array(self.ground_truth)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info("=" * 70)
        logger.info("Evaluation Results")
        logger.info("=" * 70)
        self._print_metrics(metrics)
        logger.info("=" * 70)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics"""
        if len(self.ground_truth) == 0:
            return self._empty_metrics()

        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.ground_truth, self.predictions)
        metrics['precision'] = precision_score(self.ground_truth, self.predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(self.ground_truth, self.predictions, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(self.ground_truth, self.predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(
            self.ground_truth, self.predictions, average=None, zero_division=0
        ).tolist()
        metrics['recall_per_class'] = recall_score(
            self.ground_truth, self.predictions, average=None, zero_division=0
        ).tolist()
        metrics['f1_per_class'] = f1_score(
            self.ground_truth, self.predictions, average=None, zero_division=0
        ).tolist()
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(
                self.ground_truth,
                self.probabilities[:, 1],
                average='weighted'
            )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
            metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(self.ground_truth, self.predictions, labels=[0, 1])
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            self.ground_truth,
            self.predictions,
            zero_division=0,
            output_dict=True
        )
        
        # Additional metrics
        metrics['total_samples'] = len(self.ground_truth)
        metrics['correct_predictions'] = int((self.predictions == self.ground_truth).sum())
        metrics['incorrect_predictions'] = int((self.predictions != self.ground_truth).sum())
        
        # Class distribution
        metrics['sarcasm_distribution'] = {
            'not_sarcastic': int((self.ground_truth == 0).sum()),
            'sarcastic': int((self.ground_truth == 1).sum())
        }
        
        return metrics
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics in readable format"""
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        
        if metrics['roc_auc'] is not None:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info(f"\nTotal Samples: {metrics['total_samples']}")
        logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
        logger.info(f"Incorrect Predictions: {metrics['incorrect_predictions']}")
        
        logger.info(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        logger.info(f"  True Negatives:  {cm[0, 0]}")
        logger.info(f"  False Positives: {cm[0, 1]}")
        logger.info(f"  False Negatives: {cm[1, 0]}")
        logger.info(f"  True Positives:  {cm[1, 1]}")
        
        logger.info(f"\nClass Distribution:")
        for key, val in metrics['sarcasm_distribution'].items():
            logger.info(f"  {key}: {val}")
    
    def save_results(self, output_dir: str):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)

        if len(self.ground_truth) == 0:
            logger.warning("Skipping result file generation because test set is empty.")
            return
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self._calculate_metrics(),
            'predictions': self.predictions.tolist(),
            'probabilities': self.probabilities.tolist(),
            'ground_truth': self.ground_truth.tolist(),
            'model_config': self.config['model']
        }
        
        # Save as JSON
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        save_results(results, results_file)
        logger.info(f"Results saved to {results_file}")
        
        # Save as CSV for easy viewing
        results_df = pd.DataFrame({
            'prediction': self.predictions,
            'probability_not_sarcastic': self.probabilities[:, 0],
            'probability_sarcastic': self.probabilities[:, 1],
            'ground_truth': self.ground_truth,
            'correct': self.predictions == self.ground_truth
        })
        
        results_csv = os.path.join(output_dir, 'evaluation_results.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Results CSV saved to {results_csv}")
        
        # Save detailed report
        report_file = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MULTIMODAL SARCASM DETECTION - EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            metrics = results['metrics']
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy:              {metrics['accuracy']:.4f}\n")
            f.write(f"Precision:             {metrics['precision']:.4f}\n")
            f.write(f"Recall:                {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:              {metrics['f1']:.4f}\n")
            
            if metrics['roc_auc'] is not None:
                f.write(f"ROC-AUC:               {metrics['roc_auc']:.4f}\n")
            
            f.write(f"\nTOTAL SAMPLES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total:                 {metrics['total_samples']}\n")
            f.write(f"Correct:               {metrics['correct_predictions']}\n")
            f.write(f"Incorrect:             {metrics['incorrect_predictions']}\n")
            
            f.write(f"\nCONFUSION MATRIX\n")
            f.write("-" * 70 + "\n")
            cm = np.array(metrics['confusion_matrix'])
            f.write(f"True Negatives:        {cm[0, 0]}\n")
            f.write(f"False Positives:       {cm[0, 1]}\n")
            f.write(f"False Negatives:       {cm[1, 0]}\n")
            f.write(f"True Positives:        {cm[1, 1]}\n")
            
            f.write(f"\nCLASS DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            for label, count in metrics['sarcasm_distribution'].items():
                f.write(f"{label.replace('_', ' ').title():20}: {count:5d}\n")
            
            f.write(f"\nPER-CLASS METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 70 + "\n")
            classes = ['Not Sarcastic', 'Sarcastic']
            for i, cls in enumerate(classes):
                f.write(f"{cls:<20} {metrics['precision_per_class'][i]:<12.4f} ")
                f.write(f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        logger.info(f"Detailed report saved to {report_file}")

