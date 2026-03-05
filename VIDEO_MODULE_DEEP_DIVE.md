# Video Module Deep Dive (From Scratch, Narrative Version)

This guide is written so you can teach the video module to someone else, not just read implementation notes. Instead of isolated bullets, it explains the story of what happens from the moment a media URL is found to the moment the model receives a final visual feature vector.

---

## 1) Start with the problem we are solving

Our classifier predicts sarcasm, but text alone is not always enough. A sentence may look neutral in words and still be sarcastic because of context in a frame (for example, meme format, facial expression, visual contradiction, or scene tone). The video module exists to convert visual context into numbers the model can understand.

Think of this module as a translator:

- input language: raw pixels from video or image
- output language: fixed-length vector embedding

That embedding is later fused with text and audio features.

---

## 2) Where this lives in your code

The core implementation is in src/feature_extraction.py, mainly:

- VideoFeatureExtractor
- MultimodalFeatureExtractor (visual branch)

The upstream file social_media_pipeline.py is responsible for downloading media so this module receives local files that OpenCV can read.

---

## 3) The complete lifecycle of one visual sample

Imagine one Reddit post with either video or image. The runtime flow is:

1. Social pipeline downloads media locally.
2. Multimodal extractor calls visual extractor.
3. Visual extractor samples frames.
4. Frame tensors are normalized.
5. ResNet18 trunk extracts frame-level embeddings.
6. Embeddings are pooled into one vector.
7. That vector goes to the sarcasm model with text/audio vectors.

Everything in the video module is designed so this flow is stable even when media files are broken or missing.

---

## 4) Libraries and what each one does at low level

### OpenCV (cv2)

OpenCV is your video decoder and frame navigator.

- VideoCapture(file): opens stream handle.
- CAP_PROP_FPS: reads nominal frame rate.
- CAP_PROP_FRAME_COUNT: reads total frame count.
- CAP_PROP_POS_FRAMES: jumps to an absolute frame index.
- cap.read(): decodes one frame from the current position.
- cv2.resize: spatially resizes pixel matrix.
- cv2.cvtColor(..., BGR2RGB): fixes channel order.

Why needed: without OpenCV, you cannot reliably seek and decode arbitrary frames by index.

### NumPy

NumPy manages frame arrays and vectorized math.

- linspace creates evenly spaced frame indices.
- transpose changes layout from HWC to CHW.
- float normalization and broadcasting apply mean/std efficiently.

Why needed: it is the bridge between decoded pixel buffers and tensor-ready data.

### PyTorch + torchvision

PyTorch runs the neural network. torchvision provides pretrained ResNet18.

- ResNet18 trunk converts image tensors into semantic feature vectors.
- no_grad disables gradient graph during extraction.
- tensor.to(device) manages execution placement.

Why needed: this project’s full model stack is PyTorch-based.

### PIL

Used for image fallback path. If the post has only an image, PIL loads it and the module converts it into a pseudo-frame sequence.

---

## 5) How initialization sets the module behavior

When VideoFeatureExtractor is created, it does not just “load a model.” It configures the entire contract the rest of the pipeline depends on.

It reads:

- frame_size
- n_frames_per_segment
- target_fps (stored for policy; current sampler is fixed-count)
- expected video feature dimension

Then it loads ResNet18 with ImageNet weights, removes the classification head, sets eval mode, and freezes parameters.

This means the visual encoder is used as a fixed feature generator, not retrained in your current pipeline.

---

## 6) Deep walkthrough of extract_frames

This function receives a local video path and an optional time window. Its output must always have a fixed shape, even when the video is broken.

### Step A: file existence guard

If the path does not exist, it returns zeros with shape:

(n_frames_per_segment, 3, H, W)

This prevents downstream code from crashing.

### Step B: open decoder

OpenCV handle is created. If the file cannot be opened, return zeros again.

### Step C: read metadata

It reads fps and total frame count. If either is invalid (fps <= 0 or no frames), it logs and returns zeros.

This guard was added to prevent division errors and invalid frame math.

### Step D: compute segment boundaries

If end_time is missing, the full video duration is inferred as total_frames / fps.

Then:

- start_frame = int(start_time * fps)
- end_frame = int(end_time * fps)

Both are clamped so they remain valid frame indices.

### Step E: sample frame indices

The code uses uniform index sampling via linspace over [start_frame, end_frame-1].

Important: this is fixed-count sampling, not frame-by-frame reading.

### Step F: decode each selected frame

For each sampled index:

1. seek decoder to that absolute frame
2. decode
3. if successful: resize and convert BGR to RGB

Only successfully decoded frames are appended.

### Step G: enforce fixed output length

If decoded list is empty, return zeros.

If decoded list is shorter than requested frame count, repeat the last valid frame until length matches n_frames_per_segment.

### Step H: convert memory layout

Stacked output from OpenCV is HWC per frame. The module transposes to CHW because PyTorch CNNs expect channel-first format.

Final shape from this function is always:

(F, 3, H, W)

---

## 7) Deep walkthrough of extract_features

Now the module has sampled frames. Next it creates neural features.

### Step A: convert pixel scale

Frame values are cast to float32 and divided by 255.

This maps raw 8-bit values from [0..255] into [0..1].

### Step B: ImageNet normalization

For each channel (R,G,B), it applies:

x_norm = (x - mean) / std

with ImageNet constants.

Why: pretrained ResNet18 expects this input distribution.

### Step C: convert to torch tensor

NumPy array becomes torch tensor, moved to selected device.

### Step D: forward through ResNet trunk

no_grad context is used. Output before squeeze is approximately:

(F, 512, 1, 1)

After squeezing spatial singleton dimensions:

(F, 512)

So each sampled frame becomes one 512-dimensional semantic embedding.

---

## 8) How images are handled when no video exists

The image path function creates compatibility with the same video-CNN flow.

It loads one image, resizes it, converts to CHW, and repeats it F times to build a pseudo-sequence.

Then extract_features runs exactly the same way as with video frames.

This design keeps downstream model signatures unchanged.

---

## 9) How one video turns into one vector for classifier

Inside MultimodalFeatureExtractor:

1. get frame embeddings (F,512)
2. mean-pool over frame axis
3. produce one vector (512,)

That pooled vector is what the classification model sees as visual modality input.

If no usable visual media exists, a zero vector of visual dimension is used so the model can still run with text/audio.

---

## 10) Mathematical view (for teaching)

Let sampled frames be x_1 ... x_F.

Normalize each frame:

x'_i = (x_i / 255 - mu) / sigma

Extract features with encoder g:

f_i = g(x'_i),  f_i in R^512

Pool to one video vector:

v = (1/F) * Σ f_i

This v is the visual input to multimodal fusion.

---

## 11) What makes this module production-robust

The code is fail-soft by design. Broken media does not crash inference.

Failure cases handled:

- missing file
- unreadable stream
- invalid fps/frame count
- empty decode result

In all these cases, output becomes a zero visual tensor and the rest of the pipeline continues.

This is critical for social-media data where links and formats are often unreliable.

---

## 12) Important implementation truths to teach others

1. The module is a feature extractor, not a standalone sarcasm detector.
2. ResNet18 is used, not ResNet50 (some comments may still mention ResNet50).
3. target_fps is configured but current logic uses fixed-count uniform sampling.
4. BGR->RGB conversion is mandatory; skipping it degrades feature quality.
5. The output interface is fixed shape by contract, even under media failures.

---

## 13) How to explain this in one classroom-style sentence

“Our video module samples a few representative frames from a clip, standardizes them to ImageNet format, encodes each frame with a frozen pretrained ResNet18, averages those frame embeddings into one compact visual vector, and guarantees stable output even when media files are broken.”

---

## 14) If you need to demo this live

When explaining to others, show this order:

1. media file comes in
2. frame extraction and safety guards
3. normalization and CNN encoding
4. pooling to single vector
5. fusion with text/audio in final classifier

If they ask “why not decode all frames?”, explain that fixed-count sampling is a speed/latency tradeoff for CPU inference.

---

## 15) Final takeaway

The video module in this project is a carefully engineered visual front-end that balances three goals at once: semantic quality (pretrained deep embeddings), operational speed (small fixed frame count + ResNet18), and real-world robustness (hard guards + safe zero fallbacks).

- identifies candidate video URLs
- resolves direct Reddit fallback video URL when possible
- validates `content-type`
- skips HTML/non-media responses
- enforces size limits

Without these checks, extractor would receive many non-video artifacts and fail frequently.

---

## 16) Algorithms used in video module

## 16.1 Uniform temporal sampling

Given interval `[s, e]` in frame units and `F` desired frames:

`idx = linspace(s, e-1, F)`

Complexity: O(F) seek+decode operations.

## 16.2 CNN feature extraction

Per frame tensor processed by ResNet18 trunk; output embedding dimension ≈ 512.

## 16.3 Temporal aggregation

Simple arithmetic mean over frame embeddings:

`v = (1/F) * sum_i f_i`

where `f_i` is frame embedding.

---

## 17) Mathematical formulation for visual path

Let decoded frame sequence be:

`X = {x_1, ..., x_F}`, `x_i in R^(3 x H x W)`

Normalized frame:

`x'_i = (x_i/255 - mu) / sigma`

with channel-wise `mu`, `sigma`.

Visual encoder `g` (ResNet trunk):

`f_i = g(x'_i)`, `f_i in R^512`

Pooled video embedding:

`v = (1/F) * sum_{i=1..F} f_i`

This `v` is passed to multimodal classifier.

---

## 18) Performance characteristics

Main runtime costs:

1. Video seek/decode in OpenCV.
2. CNN forward pass over sampled frames.

Because `F` is small in config (`n_frames_per_segment=3`), CPU throughput is acceptable for moderate pipeline sizes.

Potential bottlenecks:

- network download latency
- codec/container incompatibility
- repeated disk IO for large batches

---

## 19) Known implementation notes

1. Docstring in `extract_features` says “ResNet50” but implementation uses ResNet18.
2. `target_fps` is configured but not currently used for explicit FPS stepping.
3. Current temporal aggregator is mean pooling; no learned temporal attention in extractor itself.

---

## 20) Failure modes and mitigation

## 20.1 Missing file
Symptom: warning + zero visual embedding.

Mitigation: verify download path and media write permissions.

## 20.2 Invalid metadata (`fps=0`, `frames=0`)
Symptom: invalid metadata warning + zero visual embedding.

Mitigation: media file likely corrupted/non-video; inspect source URL/content-type.

## 20.3 Partial decode
Symptom: fewer frames than expected.

Mitigation: module auto-pads by repeating last valid frame.

## 20.4 No visual media
Symptom: model still predicts with text/audio (visual vector is zeros).

Mitigation: expected behavior in text-only posts.

---

## 21) Debugging checklist for video module

1. Confirm downloaded media exists and extension is valid video/image.
2. Check OpenCV can open file (`cap.isOpened`).
3. Print `fps` and `frame_count`.
4. Verify sampled indices are in valid range.
5. Inspect one decoded frame shape/dtype.
6. Confirm post-conversion shape is `(F,3,H,W)`.
7. Confirm normalized tensor range is sensible (roughly centered around 0 after normalization).
8. Confirm CNN output shape `(F,512)`.
9. Confirm pooled shape `(512,)`.

---

## 22) How this module connects to model prediction

At inference/training time:

1. Video module outputs `video_features_pooled`.
2. Text module outputs text embedding.
3. Audio module outputs audio features.
4. Model fuses all three and predicts sarcastic/not sarcastic.

So video module provides one modality contribution, not standalone sarcasm classification.

---

## 23) Suggested future improvements for visual path

1. Replace mean pooling with temporal attention or lightweight transformer.
2. Add optical-flow or motion embeddings for dynamic sarcasm cues.
3. Use stronger multimodal pretraining (e.g., CLIP-style alignment).
4. Add frame quality filtering and scene-change-aware sampling.
5. Implement explicit FPS-driven adaptive sampler using `target_fps`.

---

## 24) Summary in one sentence

The video module is a robust, CPU-friendly visual feature pipeline that samples frames from video/image inputs, converts them into normalized ResNet18 embeddings, pools to a fixed vector, and safely falls back to zero vectors when media is invalid so the multimodal sarcasm system remains stable.
