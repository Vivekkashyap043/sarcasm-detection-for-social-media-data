# Multimodal Sarcasm Detection Project Documentation (A to Z)

## 1. Executive Summary

This project is a production-oriented multimodal sarcasm detection framework built around MUSTARD++ style data and extended for real-world social media inference (especially Reddit). It combines:

- Text semantics (DistilBERT embeddings)
- Visual context (ResNet18 features from video frames or images)
- Audio/prosody cues (MFCC statistics)
- Multimodal neural fusion (LSTM / Transformer / MLP / Attention)
- Explainability (LIME, SHAP, and content-aware runtime explanations)

Primary use cases:

1. Research and benchmarking on curated sarcasm datasets.
2. End-to-end deployment-like inference on live social media posts/comments.
3. Human-readable explanations for model decisions.

---

## 2. Project Goals and Strategy

### 2.1 Problem Statement
Sarcasm is often expressed through conflict between literal wording and contextual cues. A text-only model may miss sarcasm when tone, visual scene, or delivery matters.

### 2.2 Strategy
The project follows a **multimodal late-fusion** strategy:

1. Extract per-modality dense features.
2. Align all modalities into common hidden spaces.
3. Fuse and classify into binary output: sarcastic vs not sarcastic.
4. Explain with both model-agnostic (LIME/SHAP) and runtime content-aware analysis.

### 2.3 Why this approach
- Modular: feature extractors and model backbones can be swapped independently.
- CPU-friendly: DistilBERT + ResNet18 are lighter than larger alternatives.
- Practical for noisy social data: robust fallbacks for missing/corrupt media.

---

## 3. Repository Overview

Key top-level files/folders:

- `config/config.yaml`: central runtime configuration.
- `src/`: core implementation.
- `train.py`: end-to-end training pipeline entry.
- `test.py`: evaluation/explain/predict commands.
- `social_media_pipeline.py`: Reddit ingestion + inference pipeline.
- `results/`: prediction and evaluation outputs.
- `models/`: trained checkpoints.
- `TRAINING_AND_VALIDATION.md`: operational runbook.

---

## 4. Data Layer and Preprocessing

## 4.1 Supported Metadata
Implemented in `src/data_preprocessing.py`.

Supported metadata sources:
- `metadata.xlsx` (preferred)
- `metadata.csv` (fallback)

Column normalization maps different naming styles into canonical names:
- `KEY`
- `SENTENCE`
- `Sarcasm`
- `SCENE` (generated if missing)
- `END_TIME` (defaulted if missing)

## 4.2 Label Standardization
`Sarcasm` is mapped to binary values:
- `1`: sarcastic
- `0`: not sarcastic

It supports textual variants (`true/false`, `yes/no`, `sarcastic/not sarcastic`).

## 4.3 KEY Parsing Logic
`extract_video_info()` supports:

- `1_10004_c_00` → base `1_10004_c`, type `c`, segment `0`
- `1_10004_u_00` → base `1_10004_u`, type `u`, segment `0`
- `1_10004_u` → base `1_10004_u`, type `u`, segment `0`

Fallback handling is provided for malformed keys.

## 4.4 Time Parsing
`convert_timestamp_to_seconds()` supports:
- `ss`
- `mm:ss`
- `hh:mm:ss`

## 4.5 Split Modes
Two modes from `config.yaml`:

1. **Full-data mode** (`use_full_data_for_training: true`)
   - 100% labeled data used for training
   - test split intentionally empty
2. **Standard split mode** (`use_full_data_for_training: false`)
   - stratified train/test split via `train_test_split`

---

## 5. Feature Engineering and Modalities

Implemented in `src/feature_extraction.py`.

## 5.1 Visual Features (Video/Image)

### 5.1.1 Frame Extraction
- OpenCV reads video.
- Robust guards for invalid metadata (`fps <= 0`, no frames, unreadable files).
- Uniform frame sampling (`np.linspace`) over the segment.
- Per-frame resize to configured resolution.

### 5.1.2 CNN Embedding
- Backbone: pretrained ResNet18 from torchvision.
- Final classifier head removed.
- Output per frame ≈ 512-dimensional feature.
- Temporal pooling: mean over selected frames.

### 5.1.3 Image as Pseudo-Video
A single image is resized and repeated into `n_frames_per_segment` frames for shared visual pipeline compatibility.

## 5.2 Text Features

### 5.2.1 Tokenization and Encoding
- Tokenizer/model: `distilbert-base-uncased`.
- Truncation and padding to `max_length`.
- Embedding selected from hidden state `[CLS]` token representation.

## 5.3 Audio Features

### 5.3.1 Extraction
- Uses `librosa.load()` with configured sample rate.
- Reads either direct audio file or (optional) audio stream from video container.

### 5.3.2 Features
- MFCC matrix computed.
- Mean and standard deviation per MFCC channel.
- Concatenated feature vector.
- Zero-padding or truncation to fixed `audio_feature_dim`.

### 5.3.3 Stability and Caching
- Waveform cache for repeated access.
- Failed audio paths cached to avoid repeated warnings and retries.
- Unsupported/no-audio streams safely fallback to zeros.

## 5.4 Multimodal Batch Extraction
`MultimodalFeatureExtractor.extract_batch_features()` builds tensors:

- `video_features`: `[batch, video_dim]`
- `text_features`: `[batch, text_dim]`
- `audio_features`: `[batch, audio_dim]`

---

## 6. Model Architectures

Implemented in `src/model.py`.

Supported architectures via `build_model()`:

1. `multimodal_lstm`
2. `multimodal_transformer`
3. `multimodal_mlp`
4. `multimodal_attention` (default in current config)

## 6.1 Common Design Pattern
1. Per-modality projection to hidden space.
2. Fusion operation.
3. Classification head to two logits.

## 6.2 AttentionMultimodalModel (default)
- Projects visual/text/audio into hidden vectors.
- Applies multihead self-attention over modality tokens.
- Flattens attended representation.
- MLP classification head outputs logits for 2 classes.

## 6.3 LSTM/Transformer/MLP Variants
- LSTM variant treats modality vectors as short sequences and fuses bi-LSTM outputs.
- Transformer variant uses positional embeddings over modality slots.
- MLP variant directly concatenates and feeds through fully connected blocks.

---

## 7. Training System

Implemented in `src/training.py`, orchestrated by `train.py`.

## 7.1 Training Pipeline
1. Setup reproducibility + directories.
2. Preprocess metadata and split data.
3. Initialize feature extractors + model.
4. Train with validation monitoring.
5. Save best checkpoint by validation accuracy.
6. Evaluate on test set when available.

## 7.2 Optimization
- Loss: CrossEntropyLoss
- Optimizers: Adam / AdamW / SGD
- Scheduler: CosineAnnealingLR or StepLR
- Gradient clipping: max norm = 1.0
- Early stopping by validation accuracy stagnation

## 7.3 Full-data validation strategy
When full-data mode is active:
- train on full dataset
- choose overlap validation subset for monitoring (configurable fraction)

---

## 8. Inference and Social Pipeline

## 8.1 Core Predictor
Implemented in `src/inference.py` via `MultimodalSarcasmPredictor`.

- Loads latest or specified checkpoint.
- Extracts modality features.
- Computes logits and softmax probabilities.
- Returns prediction object with confidence and explanation.

## 8.2 Modality Contribution Estimation
Ablation-style contribution estimation is used:

- Baseline confidence: full modalities
- Remove one modality at a time (replace with zero vector)
- Observe confidence drop for predicted class
- Normalize drops to contribution weights

This powers content-aware explanation fields.

## 8.3 Reddit Pipeline
Implemented in `social_media_pipeline.py`:

1. Fetch by keywords or subreddit stream.
2. Fetch comments per post.
3. Resolve/download media with type checks and size guard.
4. Prefer direct Reddit fallback video URL when available.
5. Predict post and comments.
6. Save JSON with prediction + explanation + media paths.

---

## 9. Explainability System

Implemented in `src/explainability.py`.

## 9.1 Modes
1. **Detailed mode** (`--detailed-explanations`)
   - LIME or SHAP text explainers
   - optional visual/audio/fusion explanation blocks
2. **Fast mode** (default social pipeline)
   - content-aware explanation with signal extraction + modality contributions

## 9.2 LIME safeguards
- Character-length threshold (`max_text_chars_for_lime`) to avoid runtime blowups.
- API compatibility fallback across LIME versions.

## 9.3 Content-aware explanation output
Returns fields such as:
- `summary`
- `meaning_interpretation`
- `text_signals`
- `context_signals`
- `modalities_considered`
- `modality_contributions`
- class probabilities

---

## 10. Mathematical Formulation

Let modality feature vectors be:
- visual: $v \in \mathbb{R}^{d_v}$
- text: $t \in \mathbb{R}^{d_t}$
- audio: $a \in \mathbb{R}^{d_a}$

## 10.1 Projection
$$
\tilde v = \phi_v(W_v v + b_v),\quad
\tilde t = \phi_t(W_t t + b_t),\quad
\tilde a = \phi_a(W_a a + b_a)
$$
where $\phi$ is typically ReLU + dropout block.

## 10.2 Fusion + Classification
Generic fused representation:
$$
z = f_{\text{fusion}}(\tilde v, \tilde t, \tilde a)
$$
Logits:
$$
\ell = W_c z + b_c \in \mathbb{R}^2
$$
Probabilities (softmax):
$$
p(y=k\mid x)=\frac{e^{\ell_k}}{\sum_{j=1}^{2} e^{\ell_j}}
$$

## 10.3 Loss
For binary class index $y \in \{0,1\}$:
$$
\mathcal{L}_{CE} = -\log p(y\mid x)
$$
Batch objective:
$$
\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_{CE}^{(i)}
$$

## 10.4 Metrics
Given confusion matrix terms $TP, TN, FP, FN$:
$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$
$$
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN}
$$
$$
F_1 = \frac{2\cdot \text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}
$$

## 10.5 Modality Contribution (runtime explanation)
For predicted class $\hat y$ with full confidence $p_{full}$:
$$
\Delta_m = \max\left(0,\ p_{full} - p_{-m}\right)
$$
where $p_{-m}$ is confidence after removing modality $m$.
Normalized contribution:
$$
c_m = \frac{\Delta_m}{\sum_j \Delta_j + \epsilon}
$$

## 10.6 MFCC feature summary
For MFCC matrix $M \in \mathbb{R}^{K\times T}$:
$$
\mu_k = \frac{1}{T}\sum_{t=1}^{T} M_{k,t},\quad
\sigma_k = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(M_{k,t}-\mu_k)^2}
$$
Audio vector is $[\mu_1,\ldots,\mu_K,\sigma_1,\ldots,\sigma_K]$ then padded/truncated.

---

## 11. Graphs and Visual Analytics Used

Current pipeline natively computes and/or outputs values suitable for:

1. **Confusion Matrix** (used directly in evaluation report).
2. **ROC Curve / ROC-AUC** (metrics computed using probability outputs).
3. **Precision-Recall Curve** (imports and metrics support available).
4. **Training Curves** (train/val loss and accuracy history stored in trainer).
5. **Feature Importance Bar Chart** (LIME/SHAP word contributions).
6. **Class Probability Distribution Plot** (from saved CSV/JSON probabilities).

Libraries available for graphing:
- `matplotlib`
- `seaborn`

---

## 12. Libraries and How They Work in This Project

## 12.1 Core ML/Deep Learning
- **torch / torchvision / torchaudio**: tensor ops, model definitions, pretrained backbones, training and inference.
- **transformers**: DistilBERT tokenizer + encoder for text embeddings.

## 12.2 Signal and Media Processing
- **opencv-python (cv2)**: video read/seek/resize/frame extraction.
- **Pillow**: image loading and resizing.
- **librosa + soundfile**: audio decode and MFCC extraction.

## 12.3 Data and Metrics
- **pandas / numpy**: metadata processing and numerical ops.
- **scikit-learn**: split, evaluation metrics, reports, AUC.

## 12.4 Explainability
- **lime**: local perturbation-based importance for text.
- **shap**: Shapley-value style contribution explanations.

## 12.5 Ops and Utilities
- **pyyaml**: config parsing.
- **tqdm**: progress bars.
- **requests / praw**: Reddit API and content fetching.
- **openpyxl**: `.xlsx` metadata reading.

---

## 13. End-to-End Execution Paths

## 13.1 Train
```bash
python train.py
```

## 13.2 Evaluate
```bash
python test.py test --model models/<best_model>.pth
```

## 13.3 Explain
```bash
python test.py explain --model models/<best_model>.pth --num-samples 20
```

## 13.4 Predict one sample
```bash
python test.py predict --model models/<best_model>.pth --text "example"
```

## 13.5 Social Inference
```bash
python social_media_pipeline.py
```
or
```bash
python social_media_pipeline.py --keywords "Bollywood"
```

---

## 14. Output Artifacts

- `models/best_model_*.pth`: best checkpoint
- `results/evaluation_results.json`
- `results/evaluation_results.csv`
- `results/evaluation_report.txt`
- `results/reddit_multimodal_results.json`
- `results/downloads/...` media cache

---

## 15. Error Handling and Robustness

Key resilience mechanisms:

1. Missing/invalid media returns zero feature vectors rather than crashing.
2. Invalid FPS or unreadable video metadata are handled safely.
3. Audio decoding failures are cached and skipped on reuse.
4. Empty test set is handled with safe zero metrics.
5. Explanation fallback returns valid payload even if detailed explainer fails.

---

## 16. Limitations and Future Work

Current limitations:
- Visual explanation is currently feature-summary based (not saliency maps).
- Audio explanation is heuristic placeholder in detailed mode.
- No explicit cross-lingual calibration despite multilingual social input.

Recommended upgrades:
1. Add CLIP-like joint vision-text encoders.
2. Add wav2vec2/Hubert style learned audio encoders.
3. Add calibration layer (temperature scaling).
4. Add uncertainty-aware abstention for low confidence outputs.
5. Add dedicated multilingual tokenizer/backbone.

---

## 17. Reproducibility Notes

- Set fixed random seed in config.
- Keep model/versioned dependencies pinned (`requirements.txt`).
- Store config snapshot with each checkpoint.
- Keep train/test split mode explicit in experiment logs.

---

## 18. Quick FAQ

**Q: Why is test set empty?**
A: Full-data mode is enabled for training (`use_full_data_for_training=true`).

**Q: Why do some media paths have no audio?**
A: Social media MP4 streams may not contain decodable audio track.

**Q: Why do some videos become text-only predictions?**
A: Downloader skips non-media responses and unsupported content types by design.

---

## 19. Suggested Citation (Internal)
If used in internal reports, cite as:

> Multimodal Sarcasm Detection Framework (MUSTARD++ + Social Media Extension), project repository, 2026.

---

## 20. Conclusion

This project delivers a complete multimodal sarcasm pipeline from data preprocessing to explainable social-media inference. It balances research flexibility with production robustness through modular architecture, deterministic configuration, and practical safeguards for real-world noisy inputs.
