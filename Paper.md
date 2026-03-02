# Multimodal Sarcasm Detection with Contextual Social-Media Adaptation

## Abstract
Sarcasm detection in natural language is fundamentally context-dependent and frequently underdetermined by text alone. We present a practical multimodal sarcasm detection framework that combines textual, visual, and audio cues using modular neural fusion architectures and deploys them in a noisy social-media environment. The system is trained on MUSTARD++-style annotated data and operationalized through a Reddit ingestion pipeline supporting keyword-based retrieval, media acquisition, and per-item explainable inference. Our framework supports four model backbones (LSTM, Transformer, MLP, Attention), with a default attention-based multimodal model optimized for CPU deployment. In addition to LIME/SHAP-based model explanations, we introduce a runtime content-aware explanation layer that estimates modality contributions via confidence-drop ablation. The project demonstrates an end-to-end, reproducible, and robust approach for multimodal sarcasm understanding under real-world constraints.

**Keywords:** sarcasm detection, multimodal learning, social media NLP, explainable AI, MUSTARD++, Reddit analytics

---

## 1. Introduction
Sarcasm is a figurative communication pattern in which literal surface form diverges from intended meaning. In multimodal communication, sarcasm is not solely linguistic; gesture, visual context, and prosodic cues can change interpretation. Traditional text-only sarcasm classifiers underperform when disambiguation depends on non-textual context.

This work addresses three gaps:

1. **Multimodal grounding:** integrating text, visual context, and audio/prosody.
2. **Deployment realism:** handling noisy social-media media links and malformed content.
3. **Interpretability:** producing user-facing rationale beyond scalar confidence.

We implement a practical system spanning training, evaluation, social ingestion, and explainability in one coherent framework.

---

## 2. Problem Definition
Given a multimodal sample
$$
x = (x_t, x_v, x_a)
$$
where $x_t$ is text, $x_v$ is video/image context, and $x_a$ is audio/prosody,
we learn a classifier
$$
f_\theta: x \mapsto y,\quad y \in \{0,1\}
$$
with $y=1$ indicating sarcasm.

Goal: maximize classification performance while preserving runtime robustness and interpretable outputs in social-media settings.

---

## 3. Related Approach Class
The method belongs to **late-fusion multimodal representation learning**, where each modality is encoded independently and fused at hidden level before classification. This is favored over early-fusion raw concatenation because modality-specific preprocessing and missing-modality fallbacks are easier to manage in deployment.

---

## 4. Dataset and Data Processing

## 4.1 Core Dataset
Training is based on MUSTARD++ style metadata with associated context and utterance video clips.

## 4.2 Metadata Normalization
The preprocessing module standardizes multiple metadata variants into canonical fields (`KEY`, `SENTENCE`, `Sarcasm`, optional `SCENE`, `END_TIME`) and supports both `.xlsx` and `.csv` inputs.

## 4.3 Split Strategy
Two regimes are supported:

- **Full-data training mode:** 100% labeled data for training, with overlap validation for monitoring.
- **Standard stratified split:** train/test split by sarcasm label.

## 4.4 Timestamp and Key Parsing
Video key parsing handles multiple key patterns and malformed rows using fallback logic; timestamps are converted to numeric seconds for segment-aware extraction.

---

## 5. Methodology

## 5.1 Feature Encoders

### 5.1.1 Visual Encoder
Frames are sampled and fed into a pretrained ResNet18 backbone (classification head removed), producing a fixed-dimensional visual embedding.

### 5.1.2 Text Encoder
Text is tokenized with DistilBERT tokenizer and encoded by DistilBERT. The [CLS]-level representation is used as a sentence embedding.

### 5.1.3 Audio Encoder
Audio waveform is loaded (or extracted from video if enabled), converted to MFCCs, and summarized using per-channel mean and standard deviation. The resulting vector is padded/truncated to fixed dimension.

## 5.2 Feature Projections
Let $v, t, a$ be modality vectors. They are projected to hidden space:
$$
\tilde v = \phi_v(W_v v + b_v),\quad
\tilde t = \phi_t(W_t t + b_t),\quad
\tilde a = \phi_a(W_a a + b_a)
$$
where $\phi$ denotes nonlinear projection block.

## 5.3 Fusion Architectures
Four fusion backbones are implemented:

1. **Multimodal LSTM**
2. **Multimodal Transformer**
3. **Multimodal MLP**
4. **Attention Multimodal Model** (default)

The default attention model uses multihead attention over modality tokens before classification.

## 5.4 Classification
Fused representation $z$ is mapped to logits $\ell \in \mathbb{R}^2$:
$$
\ell = W_c z + b_c
$$
Softmax probabilities:
$$
p(y=k\mid x)=\frac{e^{\ell_k}}{\sum_{j=1}^{2} e^{\ell_j}}
$$

---

## 6. Training Objective and Optimization

## 6.1 Loss
Cross-entropy loss:
$$
\mathcal{L}_{CE} = -\log p(y\mid x)
$$
Batch objective:
$$
\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{CE}^{(i)}
$$

## 6.2 Optimizer and Scheduler
Supported optimizers: Adam, AdamW, SGD.

Supported schedulers: CosineAnnealingLR, StepLR.

Gradient clipping is applied to improve numerical stability.

## 6.3 Early Stopping
Validation accuracy is monitored. Best checkpoint is persisted; training terminates when patience threshold is exceeded.

---

## 7. Evaluation Protocol
Metrics computed include:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- Per-class precision/recall/F1
- ROC-AUC (when computable)
- Confusion matrix
- Full classification report

Formulas:
$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$
$$
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN}
$$
$$
F_1 = \frac{2PR}{P+R}
$$

---

## 8. Explainability

## 8.1 Detailed Explainers
- **LIME** for local token-level text perturbation analysis.
- **SHAP** for Shapley-style token attributions.

Runtime safeguards include long-text skipping and compatibility fallback handling.

## 8.2 Content-Aware Runtime Explanation
For production/social runs, a fast explanation layer provides per-sample rationale using:

1. Text-pattern signals (markers, contrast, hyperbole, punctuation emphasis).
2. Context signals from available modalities.
3. Modality contribution estimation via confidence-drop ablation.

Contribution for modality $m$:
$$
\Delta_m = \max(0, p_{full} - p_{-m})
$$
$$
c_m = \frac{\Delta_m}{\sum_j\Delta_j + \epsilon}
$$

---

## 9. Social-Media Adaptation Pipeline
The Reddit pipeline performs:

1. Keyword/subreddit retrieval.
2. Comment retrieval by permalink.
3. Media download with content-type and size guards.
4. Direct video fallback URL resolution for Reddit-hosted media.
5. Multimodal prediction per post/comment.
6. Structured JSON output with prediction, probabilities, modality metadata, and explanation.

Robustness provisions:
- Non-media responses are skipped.
- Invalid/unreadable media yields safe zero features.
- Failed audio files are cached to prevent repetitive decode attempts.

---

## 10. Graphs and Analytical Visualizations
The system computes or stores all values needed for these plots:

1. **Confusion matrix heatmap**
2. **ROC curve** and AUC
3. **Precision-recall curve**
4. **Training curves** (loss and accuracy vs epoch)
5. **Token importance bar plots** (LIME/SHAP)
6. **Probability distribution histograms**

Graphing stack: matplotlib + seaborn.

---

## 11. Implementation Stack

### 11.1 Deep Learning and NLP
- PyTorch ecosystem (`torch`, `torchvision`, `torchaudio`)
- Hugging Face `transformers`

### 11.2 Data and Metrics
- `numpy`, `pandas`, `scikit-learn`

### 11.3 Media and Audio
- `opencv-python`, `Pillow`, `librosa`, `soundfile`

### 11.4 Explainability
- `lime`, `shap`

### 11.5 Pipeline and IO
- `requests`, `praw`, `pyyaml`, `openpyxl`, `tqdm`

---

## 12. Reproducibility and Engineering Quality
Reproducibility is supported by:

- fixed random seed
- pinned dependencies
- YAML-driven configuration
- deterministic checkpointing of best model
- explicit result artifacts in JSON/CSV/TXT formats

Engineering quality features include modular decomposition, defensive IO, and graceful fallbacks under malformed social-media data.

---

## 13. Limitations

1. Detailed visual explainability is currently feature-summary based, not spatial saliency.
2. Audio explanation in detailed mode is heuristic.
3. Domain/language shift in social data can affect calibration.
4. Real-time throughput is bounded on CPU for large-scale streaming.

---

## 14. Future Work

1. Stronger audio encoders (e.g., wav2vec2/Hubert).
2. Cross-modal pretraining (e.g., vision-language contrastive objectives).
3. Confidence calibration and selective prediction.
4. Multilingual and code-mixed text handling.
5. Richer explanation UX with per-modality confidence intervals.

---

## 15. Conclusion
This project demonstrates a practical and extensible multimodal sarcasm detection system that unifies research-grade modeling and deployment-grade robustness. By combining modular deep feature extraction, flexible fusion architectures, robust social-media ingestion, and multi-level explainability, it provides a complete framework for both experimentation and real-world inference.

---

## References (Implementation-Centric)

1. Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Sanh et al., DistilBERT, a distilled version of BERT.
3. He et al., Deep Residual Learning for Image Recognition.
4. Ribeiro et al., “Why Should I Trust You?” Explaining the Predictions of Any Classifier (LIME).
5. Lundberg and Lee, A Unified Approach to Interpreting Model Predictions (SHAP).
6. MUSTARD and MUSTARD++ benchmark lines for multimodal sarcasm datasets.
