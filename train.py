"""
Main training script for multimodal sarcasm detection
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from sklearn.model_selection import train_test_split
from utils import (
    load_config, setup_logging, seed_everything, get_device,
    create_directories, save_results, get_timestamp
)
from data_preprocessing import DataPreprocessor
from feature_extraction import MultimodalFeatureExtractor
from model import build_model
from training import SarcasmDetectionTrainer
from evaluation import SarcasmDetectionEvaluator
from text_baseline import train_text_baseline

logger = setup_logging(level="INFO")


def setup_experiment(config_path: str):
    """Setup experiment configuration"""
    config = load_config(config_path)
    
    # Set seed for reproducibility
    seed_everything(config['data']['random_seed'])
    
    # Create necessary directories
    create_directories(config['paths'])
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    return config, device


def preprocess_data(config):
    """Preprocess and split data"""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("=" * 70)
    
    preprocessor = DataPreprocessor(config)
    train_df, test_df = preprocessor.process_data()
    
    return train_df, test_df


def prepare_model_and_features(config, device):
    """Prepare model and feature extractors"""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Model and Feature Extractor Setup")
    logger.info("=" * 70)
    
    # Initialize feature extractors
    logger.info("Initializing feature extractors...")
    feature_extractor = MultimodalFeatureExtractor(config, device)
    
    # Build model
    logger.info("Building model...")
    model = build_model(config, device)
    
    logger.info(f"Model architecture: {config['model']['architecture']}")
    
    return model, feature_extractor


def train_model(
    model,
    feature_extractor,
    train_df,
    test_df,
    config,
    device
):
    """Train the model"""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 70)

    use_full_data = bool(config['data'].get('use_full_data_for_training', False))
    validation_fraction = float(config['training'].get('validation_fraction', 0.2))
    full_mode = str(config['training'].get('validation_mode_for_full_data', 'overlap')).lower()

    if use_full_data and full_mode == 'overlap':
        logger.info("Full-data mode enabled: training on 100% data with overlap validation subset")
        train_data = train_df.reset_index(drop=True)
        val_data = train_df.sample(
            frac=min(max(validation_fraction, 0.05), 0.4),
            random_state=config['data']['random_seed']
        ).reset_index(drop=True)
    else:
        # Standard split for validation
        train_data, val_data = train_test_split(
            train_df,
            test_size=validation_fraction,
            stratify=train_df['Sarcasm'],
            random_state=config['data']['random_seed']
        )
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
    
    # Initialize trainer
    trainer = SarcasmDetectionTrainer(
        model=model,
        feature_extractor=feature_extractor,
        train_data=train_data,
        val_data=val_data,
        test_data=test_df.reset_index(drop=True) if test_df is not None else None,
        config=config,
        device=device
    )
    
    # Train model
    train_results = trainer.train()
    
    return trainer.best_model_path, train_results


def evaluate_model(
    model,
    feature_extractor,
    test_df,
    config,
    device,
    best_model_path
):
    """Evaluate the model on test set"""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Model Evaluation")
    logger.info("=" * 70)
    
    # Load best model
    if best_model_path and os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize evaluator
    evaluator = SarcasmDetectionEvaluator(
        model=model,
        feature_extractor=feature_extractor,
        test_data=test_df,
        config=config,
        device=device
    )
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    # Save results
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    evaluator.save_results(results_dir)
    
    return metrics, evaluator


def main(config_path: str = "config/config.yaml", mode: str = "multimodal"):
    """Main training pipeline"""
    
    logger.info("\n" + "#" * 70)
    logger.info("# MULTIMODAL SARCASM DETECTION FRAMEWORK")
    logger.info("# Training Pipeline")
    logger.info("#" * 70)
    
    try:
        # Setup
        config, device = setup_experiment(config_path)
        
        # Preprocess data
        train_df, test_df = preprocess_data(config)

        if mode == "text_baseline":
            # Text-only baseline (often stronger than multimodal on small data)
            train_text_baseline(train_df, test_df, config)
            logger.info("Text baseline completed. See results/text_baseline_report.txt")
            return 0

        # Setup model and features
        model, feature_extractor = prepare_model_and_features(config, device)
        
        # Train model
        best_model_path, train_results = train_model(
            model, feature_extractor, train_df, test_df, config, device
        )
        
        metrics = {'accuracy': -1.0}
        if len(test_df) > 0:
            metrics, evaluator = evaluate_model(
                model, feature_extractor, test_df, config, device, best_model_path
            )
        else:
            logger.info("No test split found (full-data mode). Skipping final test-set evaluation.")
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Best validation accuracy: {train_results['best_val_acc']:.4f}")
        if metrics['accuracy'] >= 0:
            logger.info(f"Test set accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Results saved to: {config['paths']['results_dir']}")
        logger.info("=" * 70)
        
        return 0
    
    except Exception as e:
        logger.error(f"\n{'=' * 70}")
        logger.error(f"ERROR: {str(e)}")
        logger.error("=" * 70)
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multimodal sarcasm detection model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "text_baseline"],
        help="Training mode: multimodal or text_baseline"
    )
    
    args = parser.parse_args()
    
    exit_code = main(args.config, args.mode)
    sys.exit(exit_code)

