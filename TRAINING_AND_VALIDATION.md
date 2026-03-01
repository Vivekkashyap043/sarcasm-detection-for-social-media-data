# Training and Validation Guide

This guide explains how to train, validate, and run keyword-based Reddit inference with per-post classification display.

## 1) Environment Setup (CPU)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Dataset Structure (MUSTARD++)

Keep the dataset in this structure:

```
data/
├── metadata.xlsx
├── context_videos/
└── utterance_videos/
```

Notes:
- `metadata.xlsx` is preferred (CSV is also supported).
- `KEY` patterns like `1_10004_c_00`, `1_10004_u_00`, and `1_10004_u` are supported.

## 3) Training Modes

Configuration is in `config/config.yaml`.

### A) Full-data training (100%)
Default in current config:

```yaml
data:
  use_full_data_for_training: true
training:
  validation_mode_for_full_data: "overlap"
  validation_fraction: 0.2
```

Run:

```bash
python train.py
```

Behavior:
- Trains using all labeled samples.
- Uses an overlap validation subset for monitoring and early stopping.
- Skips test-set evaluation if no explicit test split exists.

### B) Standard split training + validation + test
Set in config:

```yaml
data:
  use_full_data_for_training: false
  train_ratio: 0.7
  test_ratio: 0.3
training:
  validation_fraction: 0.2
```

Run:

```bash
python train.py
```

Behavior:
- Data split to train/test first.
- Train split further split into train/validation.
- Final evaluation generated on test split.

## 4) Validation and Evaluation Commands

### Evaluate test split
```bash
python test.py test --model models/<best_model>.pth
```

### Generate explainability report
```bash
python test.py explain --model models/<best_model>.pth --num-samples 20
```

### Single-sample validation (any modality combination)
```bash
python test.py predict --model models/<best_model>.pth --text "sample text"
python test.py predict --model models/<best_model>.pth --image path\to\image.jpg
python test.py predict --model models/<best_model>.pth --video path\to\video.mp4
python test.py predict --model models/<best_model>.pth --audio path\to\audio.wav
python test.py predict --model models/<best_model>.pth --text "sample" --image path\to\image.jpg --video path\to\video.mp4
```

## 5) Keyword-based Reddit Inference (Required Flow)

The pipeline supports two modes:

### Interactive keyword prompt mode
Run:

```bash
python social_media_pipeline.py
```

Then it asks:

```text
Enter keywords separated by comma (leave empty to use subreddit mode):
```

### One-click Windows launcher

Double-click this file from Explorer or run it in terminal:

```bash
run_reddit_keyword_pipeline.bat
```

What it does:
- Activates `venv` automatically if `venv\\Scripts\\activate.bat` exists.
- Starts `social_media_pipeline.py`.
- Prompts you for keywords.
- Saves output in `results/reddit_multimodal_results.json`.

Example input:

```text
iphone launch, india vs pakistan, sarcasm
```

Pipeline behavior:
1. Fetches Reddit posts using each keyword.
2. Downloads media if present (image/video).
3. Extracts available modalities (text/image/video/audio-from-video).
4. Classifies each post/comment as `SARCASTIC` or `NOT SARCASTIC`.
5. Displays result line-by-line in console.
6. Saves complete JSON results.

### CLI keyword mode

```bash
python social_media_pipeline.py --keywords "iphone launch" "india vs pakistan" --posts-per-subreddit 10 --comments-per-post 5 --output results/reddit_multimodal_results.json
```

## 6) Per-post Classification Display

While running Reddit pipeline, console output includes lines like:

```text
[POST] r/news | <title...> -> SARCASTIC (0.912)
[COMMENT] r/news | <comment...> -> NOT SARCASTIC (0.774)
```

## 7) Output Files

- `results/reddit_multimodal_results.json`: keyword/subreddit fetched data + predictions + explanations.
- `results/evaluation_results.json`: validation/test metrics.
- `results/evaluation_report.txt`: readable evaluation report.

## 8) Troubleshooting

- If model is missing: run `python train.py` first.
- If metadata read fails: verify `openpyxl` is installed from `requirements.txt`.
- If Reddit rate-limits: reduce request volume (`--posts-per-subreddit`, `--comments-per-post`).
- If memory is high: lower `training.batch_size` in config.
