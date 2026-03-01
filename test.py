"""
Testing and prediction script for multimodal sarcasm detection
"""
import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from utils import load_config, setup_logging, get_device
from feature_extraction import MultimodalFeatureExtractor
from model import build_model
from evaluation import SarcasmDetectionEvaluator
from explainability import SarcasmExplainer, simple_explain_prediction

logger = setup_logging(level="INFO")


def load_trained_model(model_path: str, config, device):
    """Load trained model"""
    logger.info(f"Loading model from: {model_path}")
    
    # Build model
    model = build_model(config, device)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_sample(
    text: Optional[str],
    video_path: Optional[str],
    image_path: Optional[str],
    audio_path: Optional[str],
    model,
    feature_extractor,
    config,
    device,
    start_time: float = 0.0,
    end_time: float = None
):
    """Make prediction for a single sample"""
    
    model.eval()
    
    with torch.no_grad():
        # Extract features
        video_feat, text_feat, audio_feat = feature_extractor.extract_multimodal_features(
            video_path=video_path,
            text=text,
            start_time=start_time,
            end_time=end_time,
            image_path=image_path,
            audio_path=audio_path
        )
        
        # Add batch dimension
        video_feat = video_feat.unsqueeze(0)
        text_feat = text_feat.unsqueeze(0)
        audio_feat = audio_feat.unsqueeze(0)
        
        # Forward pass
        outputs = model(video_feat, text_feat, audio_feat)
        probs = torch.softmax(outputs, dim=1)
        
        pred_label = outputs.argmax(dim=1).item()
        pred_prob = probs[0, pred_label].item()
        confidence = probs[0].cpu().numpy()
    
    return {
        'prediction': pred_label,
        'label_name': 'SARCASTIC' if pred_label else 'NOT SARCASTIC',
        'confidence': float(pred_prob),
        'probabilities': {
            'not_sarcastic': float(confidence[0]),
            'sarcastic': float(confidence[1])
        }
    }


def test_on_test_set(model_path: str, config_path: str = "config/config.yaml"):
    """Run comprehensive test on test set"""
    
    logger.info("\n" + "=" * 70)
    logger.info("Testing on Test Set")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config(config_path)
    device = get_device(config)
    
    # Load model
    model = load_trained_model(model_path, config, device)
    
    # Initialize feature extractor
    feature_extractor = MultimodalFeatureExtractor(config, device)
    
    # Load test data
    test_data_path = os.path.join(config['data']['processed_data_path'], 'test', 'metadata.csv')
    test_df = pd.read_csv(test_data_path)
    
    logger.info(f"Test set size: {len(test_df)}")

    if len(test_df) == 0:
        logger.warning(
            "Test set is empty. This happens when use_full_data_for_training=true. "
            "Disable full-data mode or create a separate holdout set for test evaluation."
        )
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'total_samples': 0
        }
    
    # Evaluate
    evaluator = SarcasmDetectionEvaluator(
        model=model,
        feature_extractor=feature_extractor,
        test_data=test_df,
        config=config,
        device=device
    )
    
    metrics = evaluator.evaluate()
    evaluator.save_results(config['paths']['results_dir'])
    
    return metrics


def generate_explanations(model_path: str, config_path: str = "config/config.yaml", num_samples: int = 20):
    """Generate explanations for test predictions"""
    
    logger.info("\n" + "=" * 70)
    logger.info("Generating Explanations")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config(config_path)
    device = get_device(config)
    
    # Load model
    model = load_trained_model(model_path, config, device)
    
    # Initialize feature extractor and explainer
    feature_extractor = MultimodalFeatureExtractor(config, device)
    explainer = SarcasmExplainer(model, feature_extractor, config, device)
    
    # Load test data
    test_data_path = os.path.join(config['data']['processed_data_path'], 'test', 'metadata.csv')
    test_df = pd.read_csv(test_data_path)
    
    # Generate explanations
    explanations_dir = os.path.join(config['paths']['results_dir'], 'explanations')
    explanations = explainer.generate_explanations_batch(
        test_df,
        explanations_dir,
        num_samples=num_samples
    )
    
    return explanations


def predict_custom_sample(
    text: Optional[str],
    video_path: Optional[str],
    image_path: Optional[str],
    audio_path: Optional[str],
    model_path: str,
    config_path: str = "config/config.yaml"
):
    """Make prediction on custom sample"""
    
    # Load configuration
    config = load_config(config_path)
    device = get_device(config)
    
    # Load model
    model = load_trained_model(model_path, config, device)
    
    # Initialize feature extractor
    feature_extractor = MultimodalFeatureExtractor(config, device)
    
    # Make prediction
    result = predict_sample(text, video_path, image_path, audio_path, model, feature_extractor, config, device)
    
    return result


def main():
    """Main testing pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Test and make predictions with multimodal sarcasm detection model"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test on test set')
    test_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    test_parser.add_argument(
        '--config',
        type=str,
        default="config/config.yaml",
        help='Path to configuration file'
    )
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Generate explanations')
    explain_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    explain_parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of samples to explain'
    )
    explain_parser.add_argument(
        '--config',
        type=str,
        default="config/config.yaml",
        help='Path to configuration file'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction on custom sample')
    predict_parser.add_argument(
        '--text',
        type=str,
        required=False,
        help='Text to predict'
    )
    predict_parser.add_argument(
        '--video',
        type=str,
        required=False,
        help='Path to video file'
    )
    predict_parser.add_argument(
        '--image',
        type=str,
        required=False,
        help='Path to image file'
    )
    predict_parser.add_argument(
        '--audio',
        type=str,
        required=False,
        help='Path to audio file (.wav/.mp3)'
    )
    predict_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    predict_parser.add_argument(
        '--config',
        type=str,
        default="config/config.yaml",
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    if args.command == 'test':
        metrics = test_on_test_set(args.model, args.config)
        logger.info(f"\nTest Results: {metrics.get('accuracy', 0.0):.4f}")
        return 0
    
    elif args.command == 'explain':
        explanations = generate_explanations(args.model, args.config, args.num_samples)
        logger.info(f"\nGenerated {len(explanations)} explanations")
        return 0
    
    elif args.command == 'predict':
        if not any([args.text, args.video, args.image, args.audio]):
            logger.error("Provide at least one of --text, --video, --image, or --audio")
            return 1

        if args.video and not os.path.exists(args.video):
            logger.error(f"Video file not found: {args.video}")
            return 1

        if args.image and not os.path.exists(args.image):
            logger.error(f"Image file not found: {args.image}")
            return 1

        if args.audio and not os.path.exists(args.audio):
            logger.error(f"Audio file not found: {args.audio}")
            return 1
        
        result = predict_custom_sample(args.text, args.video, args.image, args.audio, args.model, args.config)
        
        logger.info("\n" + "=" * 70)
        logger.info("PREDICTION RESULT")
        logger.info("=" * 70)
        if args.text:
            logger.info(f"Input Text: {args.text}")
        logger.info(f"Prediction: {result['label_name']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        logger.info(f"Probabilities: {result['probabilities']}")
        
        # Simple explanation
        if args.text:
            explanation = simple_explain_prediction(
                args.text,
                result['prediction']
            )
            logger.info(f"\n{explanation}")
        logger.info("=" * 70)
        
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

