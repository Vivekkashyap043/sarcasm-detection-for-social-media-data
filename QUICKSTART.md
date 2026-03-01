# Quick Start Guide

A quick reference for getting up and running with the Multimodal Sarcasm Detection Framework.

## ⚡ 5-Minute Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your data structure matches:
```
data/
├── metadata.xlsx          # Your annotation file (metadata.csv also supported)
├── context_videos/        # Context video files
└── utterance_videos/      # Utterance video files
```

### 3. Run Training
```bash
# Start training with default settings
python train.py

# Monitor progress in console output
```

### 4. Evaluate Results
```bash
# Find your best model in models/ directory
# Run evaluation on test set
python test.py test --model models/best_model_YYYYMMDD_HHMMSS_epochX.pth

# View results in results/ directory
```

## 🎯 Common Commands

### Training Only
```bash
python train.py --config config/config.yaml
```

### Text-Only Baseline (often higher accuracy on small data)
```bash
python train.py --mode text_baseline
```

### Testing Trained Model
```bash
python test.py test --model models/best_model.pth
```

### Generate Explanations (why is it sarcastic?)
```bash
python test.py explain \
    --model models/best_model.pth \
    --num-samples 20
```

### Predict on Custom Sample
```bash
python test.py predict \
    --model models/best_model.pth \
  --text "Your text here" \
  --video path/to/video.mp4 \
  --audio path/to/audio.wav

# Predict from Reddit with media download + explainability
python social_media_pipeline.py \
  --subreddits AskReddit funny programming \
  --posts-per-subreddit 10 \
  --comments-per-post 5 \
  --output results/reddit_multimodal_results.json
```

## 📊 Results Interpretation

After testing, check these files:

| File | Purpose |
|------|---------|
| `results/evaluation_results.json` | Complete metrics in JSON |
| `results/evaluation_results.csv` | Raw predictions |
| `results/evaluation_report.txt` | Human-readable report |
| `results/explanations/` | Why predictions were made |

## 🔧 Customization

### Modify Configuration
Edit `config/config.yaml`:

```yaml
# Faster training (but lower accuracy)
training:
  batch_size: 16          # Increase for faster batches
  num_epochs: 10          # Reduce epochs
  learning_rate: 5e-3     # Increase learning rate

# Better quality (but slower)
training:
  batch_size: 4           # Smaller batches
  num_epochs: 30          # More epochs
  learning_rate: 1e-3     # Lower learning rate
```

### Try Different Model Architecture
```yaml
model:
  architecture: "multimodal_transformer"  # Options: multimodal_lstm, multimodal_transformer, multimodal_mlp, multimodal_attention
```

## 📈 Performance Tips

### To Improve Accuracy
1. Increase `num_epochs` in config
2. Decrease `batch_size`
3. Use `multimodal_transformer` architecture
4. Reduce `learning_rate`
5. Add more training data

### To Speed Up Training
1. Increase `batch_size`
2. Use `multimodal_mlp` architecture
3. Increase `target_fps` (extract fewer frames)
4. Reduce `num_epochs`

### For CPU Deployment
1. Use `multimodal_mlp` architecture
2. Set `batch_size: 8` or lower
3. Use `target_fps: 2` for videos
4. Enable quantization if needed

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not found | Set `device_type: cpu` in config |
| Out of Memory | Reduce `batch_size` to 4 or 2 |
| Slow training | Increase `batch_size`, use MLP model |
| Model not training | Lower `learning_rate`, check data |
| Videos not found | Check file paths and naming |
| Excel read error | Install dependencies from `requirements.txt` (includes `openpyxl`) |
| Accuracy too low | Increase `num_epochs`, use Transformer |

## 📚 Project Structure Quick Reference

```
Your working directory structure:
├── train.py              ← Run this to train
├── test.py               ← Run this to test
├── config.yaml           ← Modify this for settings
└── data/                 ← Your MUSTARD++ data here
    ├── metadata.csv
    ├── context_videos/
    └── utterance_videos/
```

## 🚀 Next Steps

1. **Prepare Data**: Ensure all videos are present
2. **Configure**: Adjust `config/config.yaml` as needed  
3. **Train**: Run `python train.py`
4. **Evaluate**: Run `python test.py test --model models/best_model.pth`
5. **Analyze**: Open results files in `results/` directory
6. **Explain**: Generate explanations with `python test.py explain`
7. **Deploy**: Use best model for predictions on new data

## 💡 Pro Tips

- Check `logs/training.log` for detailed training information
- Save model checkpoints for later use or comparison
- Use different configs for different experiments
- Profile code performance if needed: `python -m cProfile train.py`
- Use git to track configuration and code changes

## 📞 Support

For questions or issues:
1. Check README.md for detailed documentation
2. Review evaluation_report.txt for metric explanations
3. Check logs/training.log for error messages
4. Search existing GitHub issues
5. Create new issue with full error trace

---

**Happy Sarcasm Detection! 🎭**

