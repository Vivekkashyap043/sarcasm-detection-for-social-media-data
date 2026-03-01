# Multimodal Sarcasm Detection Framework

A production-grade framework for detecting sarcasm in videos using both visual and textual modalities. Built with PyTorch, TensorFlow, and optimized for CPU inference.

## Features

- **Multimodal Analysis**: Combines video and text features for robust sarcasm detection
- **Multiple Architectures**: LSTM, Transformer, MLP, and Attention-based models
- **Explainability**: LIME and SHAP integration for interpretable predictions
- **CPU Optimized**: Designed to work efficiently on CPU-only systems
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and reports
- **Production Ready**: Industry-standard code structure and best practices

## Project Structure

```
sarcasm-detection/
├── config/
│   └── config.yaml                 # Configuration file for all settings
├── data/
│   ├── metadata.xlsx              # Dataset metadata (metadata.csv also supported)
│   ├── context_videos/            # Context video clips
│   ├── utterance_videos/          # Utterance video clips
│   └── processed/                 # Processed data splits
│       ├── train/
│       └── test/
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Utility functions
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── feature_extraction.py      # Video and text feature extraction
│   ├── model.py                   # Model architectures
│   ├── training.py                # Training logic
│   ├── evaluation.py              # Evaluation metrics
│   └── explainability.py          # LIME and SHAP explanations
├── models/                        # Saved trained models
├── results/                       # Test results and metrics
├── logs/                          # Training logs
├── train.py                       # Main training script
├── test.py                        # Testing and prediction script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### 1. Clone or setup the project
```bash
cd sarcasm-detection
```

### 2. Create virtual environment
```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt includes CPU-specific PyTorch builds. If you have GPU support, update the PyTorch installation:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Prepare the dataset

The MUSTARD++ dataset should have the following structure:
```
data/
├── metadata.xlsx                  # Annotation file (metadata.csv also supported)
├── context_videos/               # Context clips (1_10004_c.mp4, etc.)
└── utterance_videos/             # Utterance clips (1_10004_u.mp4, etc.)
```

## Configuration

Edit `config/config.yaml` to customize:

- **Data settings**: train/test split ratio, random seed
- **Video processing**: frame extraction rate, resolution
- **Text processing**: tokenizer model, max length
- **Model architecture**: network type, hidden dimensions, dropout
- **Training**: batch size, learning rate, epochs, optimizer
- **Explainability**: explanation method (LIME/SHAP)

Detailed workflow docs:
- `TRAINING_AND_VALIDATION.md` for training/validation + keyword-driven Reddit inference

## Training

### Start training with default configuration:
```bash
python train.py
```

### Use custom configuration:
```bash
python train.py --config path/to/custom_config.yaml
```

The training pipeline includes:
1. **Data Preprocessing**: Supports `metadata.xlsx` and full-data training mode (`use_full_data_for_training: true`)
2. **Feature Extraction**: Video frames and text tokenization
3. **Model Training**: With validation and early stopping
4. **Model Evaluation**: Comprehensive metrics on test set

### Training Progress
- Training logs appear in console and `logs/training.log`
- Models are saved to `models/` directory
- Best model is automatically selected based on validation accuracy

## Testing and Evaluation

### Test on the test set:
```bash
python test.py test --model models/best_model.pth
```

### Generate explanations for predictions:
```bash
python test.py explain --model models/best_model.pth --num-samples 20
```

### Make prediction on custom sample:
```bash
python test.py predict \
    --model models/best_model.pth \
    --text "Your sarcastic text here" \
  --video path/to/video.mp4 \
  --audio path/to/audio.wav

python social_media_pipeline.py \
  --keywords "iphone launch" "india vs pakistan" \
  --posts-per-subreddit 10 \
  --comments-per-post 5 \
  --output results/reddit_multimodal_results.json

python social_media_pipeline.py \
  --subreddits AskReddit funny programming \
  --posts-per-subreddit 10 \
  --comments-per-post 5 \
  --output results/reddit_multimodal_results.json

# Interactive mode (asks for keywords first)
python social_media_pipeline.py

# Windows one-click launcher (asks keywords interactively)
run_reddit_keyword_pipeline.bat
```

## Results and Metrics

### Test Results Location
Results are saved in `results/` directory with files:
- `evaluation_results.json`: Complete metrics in JSON format
- `evaluation_results.csv`: Predictions and probabilities
- `evaluation_report.txt`: Formatted report
- `explanations/`: Detailed explanations for sample predictions

### Key Metrics

The evaluation provides:
- **Accuracy**: Overall correctness of predictions
- **Precision**: False positive rate control
- **Recall**: False negative rate control
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Breakdown of prediction types
- **Per-class metrics**: Performance for each class
- **Classification Report**: Detailed breakdown by class

### Example Results Output
```
==============================================================================
Accuracy:  0.8234
Precision: 0.8156
Recall:    0.8234
F1-Score:  0.8189
ROC-AUC:   0.8956

Total Samples: 1200
Correct Predictions: 988
Incorrect Predictions: 212

Confusion Matrix:
  True Negatives:  450
  False Positives: 75
  False Negatives: 87
  True Positives:  588
```

## Explainability

The framework provides two explanation methods:

### 1. LIME (Local Interpretable Model-agnostic Explanations)
Shows which words contribute most to the prediction:
```
Top Contributing Words:
  'great'      :  0.3452
  'sure'       :  0.2891
  'perfect'    :  0.1234
```

### 2. Simple Explanations
Human-readable explanations for non-technical users:
```
This statement is SARCASTIC because:
1. Contains sarcasm indicators: great, sure
2. The tone appears to convey opposite meaning from literal words
3. Context suggests ironic or mocking intention
```

## Model Architectures

### 1. MultimodalLSTMModel
- Bidirectional LSTM encoding for each modality
- Fusion layer combining both modalities
- Good for sequential data

### 2. MultimodalTransformerModel
- Transformer encoder with multi-head attention
- Cross-modality attention mechanism
- Excellent for capturing long-range dependencies

### 3. MultimodalMLPModel
- Simple multi-layer perceptron
- Fastest training, minimal parameters
- Good baseline model

### 4. AttentionMultimodalModel
- Explicit attention mechanism
- Modality-specific attention weights
- Interpretable attention patterns

## Performance Optimization

### CPU Optimization Tips
1. **Reduce batch size**: Use smaller batches (4-8) for CPU
2. **Lower frame rate**: Extract 2 FPS instead of 30 FPS
3. **Smaller model**: Use DistilBERT instead of full BERT
4. **Fewer layers**: Use 2 LSTM/Transformer layers instead of 4+

### Memory Management
- Process videos in smaller chunks
- Use gradient checkpointing for large models
- Clear cache between batches

## Extending the Framework

### Adding Custom Architectures
1. Create new model class in `src/model.py`
2. Implement `forward()` method
3. Update `build_model()` function
4. Add configuration to `config.yaml`

### Adding New Features
1. Extend `MultimodalFeatureExtractor` in `src/feature_extraction.py`
2. Add audio features, object detection, sentiment analysis, etc.
3. Update feature dimensions in config

### Custom Datasets
1. Preprocess to match metadata.csv format
2. Update `DataPreprocessor` if needed
3. Ensure video files match expected naming convention

## Troubleshooting

### Common Issues

**Issue**: Out of memory error
- Solution: Reduce batch size in config.yaml

**Issue**: Video file not found
- Solution: Ensure video files are in correct directories with expected naming

**Issue**: Model not improving
- Solution: Adjust learning rate, try different architecture, increase training epochs

**Issue**: Slow training
- Solution: This is normal on CPU. Consider using smaller models or GPU

## Citation

If you use this framework, please cite:

```bibtex
@software{sarcasm_detection_2024,
  author = {Your Name},
  title = {Multimodal Sarcasm Detection Framework},
  year = {2024},
  url = {https://github.com/yourprofile/sarcasm-detection}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact the development team.

## Acknowledgments

- MUSTARD++ dataset creators and maintainers
- PyTorch and TensorFlow communities
- LIME and SHAP developers

---

**Last Updated**: 2024
**Version**: 1.0.0

