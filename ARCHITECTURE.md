# Project Architecture and API Reference

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│         Multimodal Sarcasm Detection Framework                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Video Input     │         │  Text Input      │             │
│  │  (MP4/AVI)       │         │  (Transcripts)   │             │
│  └────────┬─────────┘         └────────┬─────────┘             │
│           │                            │                       │
│           └─────────────┬──────────────┘                       │
│                         │                                      │
│              ┌──────────▼──────────┐                         │
│              │ Feature Extraction   │                         │
│              │  • Video Frames      │                         │
│              │  • BERT Embeddings   │                         │
│              └──────────┬──────────┘                         │
│                         │                                      │
│              ┌──────────▼──────────┐                         │
│              │ Feature Fusion      │                         │
│              │  • Concatenate      │                         │
│              │  • Project to space │                         │
│              └──────────┬──────────┘                         │
│                         │                                      │
│       ┌─────────────────▼─────────────────┐                 │
│       │  Multimodal Classification Model  │                 │
│       │  (LSTM/Transformer/MLP/Attention)│                 │
│       └─────────────────┬─────────────────┘                 │
│                         │                                      │
│              ┌──────────▼──────────┐                         │
│              │ Classification Head │                         │
│              │  [Sarcastic / Not]  │                         │
│              └──────────┬──────────┘                         │
│                         │                                      │
│        ┌────────────────┴────────────────┐                  │
│        │                                 │                   │
│   ┌────▼────┐                   ┌───────▼───────┐         │
│   │Prediction │                  │ Explanation   │         │
│   │[0.85]    │                  │ (LIME/SHAP)  │         │
│   └──────────┘                   └───────────────┘         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules

#### 1. `utils.py`
Utility functions and helper classes.

**Key Classes**:
- `AverageMeter`: Running average calculator
- `setup_logging()`: Configure logger

**Key Functions**:
- `load_config()`: Load YAML configuration
- `save_results()`: Save results to JSON
- `seed_everything()`: Set random seeds
- `get_device()`: Get torch device
- `count_parameters()`: Count model parameters

#### 2. `data_preprocessing.py`
Data loading, preprocessing, and splitting.

**Key Classes**:
- `DataPreprocessor`: Main preprocessing class

**Key Methods**:
- `load_metadata()`: Load CSV metadata
- `clean_metadata()`: Clean and validate data
- `split_train_test()`: 70/30 stratified split
- `extract_video_info()`: Parse video information from keys
- `process_data()`: Full pipeline

#### 3. `feature_extraction.py`
Extract visual and textual features.

**Key Classes**:
- `VideoFeatureExtractor`: Extract video features using ResNet50
- `TextFeatureExtractor`: Extract text features using DistilBERT
- `MultimodalFeatureExtractor`: Combined extractor
- `FeatureFusion`: Combine features (not used directly)

**Key Methods**:
- `extract_frames()`: Extract video frames
- `extract_features()`: Get CNN/BERT embeddings
- `extract_batch_features()`: Process multiple samples

#### 4. `model.py`
Model architectures for classification.

**Key Classes**:
- `MultimodalLSTMModel`: LSTM-based architecture
- `MultimodalTransformerModel`: Transformer-based architecture
- `MultimodalMLPModel`: Simple MLP architecture
- `AttentionMultimodalModel`: Attention-based architecture

**Key Function**:
- `build_model()`: Factory function to create model from config

#### 5. `training.py`
Training loop and logic.

**Key Classes**:
- `SarcasmDetectionTrainer`: Main trainer class
- `TrainerConfig`: Training configuration

**Key Methods**:
- `train()`: Main training loop
- `train_epoch()`: Single epoch training
- `validate()`: Validation step
- `_save_checkpoint()`: Save model

#### 6. `evaluation.py`
Evaluation and metrics calculation.

**Key Classes**:
- `SarcasmDetectionEvaluator`: Evaluation class

**Key Methods**:
- `evaluate()`: Run evaluation
- `_calculate_metrics()`: Compute all metrics
- `save_results()`: Save results to files

#### 7. `explainability.py`
Generate explanations for predictions.

**Key Classes**:
- `SarcasmExplainer`: Explanation generator

**Key Methods**:
- `explain_prediction()`: Explain single prediction
- `generate_explanations_batch()`: Batch explanations
- `_explain_with_lime()`: LIME explanations
- `_explain_with_shap()`: SHAP explanations

**Key Function**:
- `simple_explain_prediction()`: Human-readable explanation

## API Reference

### Feature Extraction API

#### VideoFeatureExtractor

```python
from feature_extraction import VideoFeatureExtractor

extractor = VideoFeatureExtractor(config, device)

# Extract frames from video
frames = extractor.extract_frames(
    video_path="path/to/video.mp4",
    start_time=0.0,
    end_time=30.0
)  # Returns: (n_frames, 3, 224, 224)

# Extract CNN features
features = extractor.extract_features(frames)  # Returns: (n_frames, 2048)
```

#### TextFeatureExtractor

```python
from feature_extraction import TextFeatureExtractor

extractor = TextFeatureExtractor(config, device)

# Extract BERT embeddings
text_features = extractor.extract_features(
    text="This movie was absolutely great"
)  # Returns: (768,)
```

#### MultimodalFeatureExtractor

```python
from feature_extraction import MultimodalFeatureExtractor

extractor = MultimodalFeatureExtractor(config, device)

# Extract both modalities
video_feat, text_feat = extractor.extract_video_text_features(
    video_path="context.mp4",
    text="Some dialogue",
    start_time=0.0,
    end_time=30.0
)  # Returns: (2048,), (768,)
```

### Model API

#### Building Models

```python
from model import build_model

model = build_model(config, device)

# Forward pass
output = model(video_features, text_features)
# video_features: (batch_size, 2048)
# text_features: (batch_size, 768)
# output: (batch_size, 2) -> logits for 2 classes

# Get probabilities
probs = torch.softmax(output, dim=1)
```

### Training API

#### Training a Model

```python
from training import SarcasmDetectionTrainer

trainer = SarcasmDetectionTrainer(
    model=model,
    feature_extractor=feature_extractor,
    train_data=train_df,
    val_data=val_df,
    config=config,
    device=device
)

results = trainer.train()
# Returns: {
#   'best_val_acc': float,
#   'training_time': float,
#   'history': dict,
#   'best_model_path': str
# }
```

### Evaluation API

#### Evaluating Model

```python
from evaluation import SarcasmDetectionEvaluator

evaluator = SarcasmDetectionEvaluator(
    model=model,
    feature_extractor=feature_extractor,
    test_data=test_df,
    config=config,
    device=device
)

metrics = evaluator.evaluate()
# Returns: {
#   'accuracy': float,
#   'precision': float,
#   'recall': float,
#   'f1': float,
#   'roc_auc': float,
#   'confusion_matrix': list,
#   'classification_report': dict,
#   ...
# }

evaluator.save_results('results/')
```

### Explainability API

#### Explaining Predictions

```python
from explainability import SarcasmExplainer

explainer = SarcasmExplainer(model, feature_extractor, config, device)

explanation = explainer.explain_prediction(
    text="Your sarcastic text",
    video_path="path/to/video.mp4",
    ground_truth_label=1
)
# Returns: {
#   'text': str,
#   'predicted_label': int,
#   'predicted_probability': float,
#   'ground_truth_label': int,
#   'is_correct': bool,
#   'text_explanation': {
#     'method': str,
#     'feature_importance': dict
#   }
# }

# Batch explanations
explanations = explainer.generate_explanations_batch(
    test_data=test_df,
    output_dir='results/explanations',
    num_samples=20
)
```

## Data Formats

### Input Data

#### Video Features
```python
# Shape: (batch_size, 2048)
# Values: Float32, normalized CNN activations
# Generated by: VideoFeatureExtractor using ResNet50
```

#### Text Features
```python
# Shape: (batch_size, 768)
# Values: Float32, BERT sentence embeddings
# Generated by: TextFeatureExtractor using DistilBERT
```

### Output Format

#### Model Output
```python
# Shape: (batch_size, 2)
# Index 0: Not Sarcastic logit
# Index 1: Sarcastic logit
# Apply softmax for probabilities
```

#### Metrics Dictionary
```python
{
    'accuracy': 0.85,
    'precision': 0.83,
    'recall': 0.87,
    'f1': 0.85,
    'roc_auc': 0.92,
    'confusion_matrix': [[450, 75], [87, 588]],
    'classification_report': {
        '0': {'precision': 0.84, 'recall': 0.86, ...},
        '1': {'precision': 0.89, 'recall': 0.87, ...}
    }
}
```

## Configuration Schema

```yaml
data:
  train_ratio: 0.7          # Training set ratio
  test_ratio: 0.3           # Test set ratio
  random_seed: 42           # Reproducibility seed

video:
  target_fps: 2             # Frames per second to extract
  frame_size: [224, 224]    # CNN input size
  n_frames_per_segment: 3   # Frames per video segment

text:
  max_length: 256           # Max tokens
  model_name: "distilbert-base-uncased"  # Pre-trained model

model:
  architecture: "multimodal_lstm"  # Architecture choice
  hidden_dim: 512           # Hidden layer dimension
  num_layers: 2             # Number of layers
  dropout: 0.3              # Dropout rate
  output_dim: 2             # Number of classes

training:
  batch_size: 8             # Batch size
  learning_rate: 1e-3       # Learning rate
  num_epochs: 20            # Number of epochs
  optimizer: "adam"         # Optimizer
  early_stopping_patience: 5  # Early stopping patience

device:
  device_type: "cpu"        # CPU or cuda
```

## Performance Characteristics

### Model Sizes (Parameters)

| Architecture | Parameters | GPU Memory | CPU Memory |
|-------------|-----------|-----------|-----------|
| MLP | ~200K | ~100MB | ~150MB |
| LSTM | ~900K | ~300MB | ~500MB |
| Transformer | ~1.2M | ~400MB | ~600MB |
| Attention | ~800K | ~300MB | ~400MB |

### Speed (Inference time per sample, CPU)

| Batch Size | LSTM | Transformer | MLP |
|-----------|------|-------------|-----|
| 1 | ~2.5s | ~2.8s | ~0.3s |
| 8 | ~1.8s | ~2.1s | ~0.12s |
| 16 | ~1.5s | ~1.8s | ~0.09s |

*Note: Includes video frame extraction*

## Extension Points

### Add Custom Architecture

1. Inherit from `nn.Module`
2. Implement `forward(video_features, text_features)`
3. Add to `build_model()` function

```python
class CustomModel(nn.Module):
    def __init__(self, video_dim, text_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        # Your implementation
    
    def forward(self, video_features, text_features):
        # Your implementation
        return logits  # (batch_size, num_classes)
```

### Add Custom Feature Extractor

1. Add method to `MultimodalFeatureExtractor`
2. Return tensor of shape `(batch_size, feature_dim)`

```python
def extract_audio_features(self, audio_path):
    # Your audio processing
    return audio_features  # (batch_size, audio_dim)
```

### Add Custom Evaluation Metric

1. Add method to `SarcasmDetectionEvaluator`
2. Update `_calculate_metrics()` to include new metric

```python
def custom_metric(self):
    # Calculate your metric
    return metric_value
```

---

For more information, refer to the [README.md](README.md) and source code documentation.

