# Social Media Data Integration Guide

Guide for collecting and preparing social media data (Reddit, Twitter, etc.) for sarcasm detection.

## Overview

Once your model is trained on the MUSTARD++ dataset, you can use it to classify sarcasm in social media content. This guide explains:

1. How to collect social media data
2. How to prepare different data modalities
3. How to use the trained model for prediction
4. How to explain predictions to users

## Supported Data Modalities

The model can handle various input combinations:

| Input Type | Support | Notes |
|-----------|---------|-------|
| Text only | ✓ | Provide dummy video path |
| Video only | ✓ | Empty text field |
| Image only | ✓ | Convert to video frames or use as context |
| Text + Video | ✓ | Native multimodal support |
| Text + Image | ✓ | Image frames extracted |
| Video + Image | ✓ | Combined visual features |
| Text + Video + Image | ✓ | Full multimodal |

## Collecting Social Media Data

### Reddit Data

#### Using PRAW (Python Reddit API Wrapper)

```python
import praw
import csv
from datetime import datetime

# Initialize Reddit API
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='SarcasmDetector/1.0'
)

def collect_reddit_comments(subreddit_name: str, limit: int = 1000):
    """Collect comments from subreddit"""
    
    subreddit = reddit.subreddit(subreddit_name)
    comments_data = []
    
    for submission in subreddit.hot(limit=limit):
        submission.comments.replace_more(limit=0)
        
        for comment in submission.comments.list():
            comments_data.append({
                'platform': 'Reddit',
                'author': comment.author.name if comment.author else 'DELETED',
                'text': comment.body,
                'subreddit': subreddit_name,
                'score': comment.score,
                'timestamp': datetime.fromtimestamp(comment.created_utc),
                'url': comment.permalink
            })
    
    return comments_data

# Example usage
comments = collect_reddit_comments('funny', limit=500)

# Save to CSV
with open('social_media_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['platform', 'author', 'text', 'subreddit', 'score', 'timestamp', 'url'])
    writer.writeheader()
    writer.writerows(comments)
```

### Twitter/X Data

```python
import tweepy
import pandas as pd

# Initialize Twitter API
client = tweepy.Client(bearer_token='YOUR_BEARER_TOKEN')

def collect_tweets(query: str, max_results: int = 100):
    """Collect tweets based on search query"""
    
    tweets_data = []
    
    response = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=['created_at', 'author_id', 'public_metrics']
    )
    
    for tweet in response.data:
        tweets_data.append({
            'platform': 'Twitter',
            'text': tweet.text,
            'created_at': tweet.created_at,
            'likes': tweet.public_metrics['like_count'],
            'retweets': tweet.public_metrics['retweet_count']
        })
    
    return tweets_data

# Example usage
tweets = collect_tweets('query here', max_results=100)
df = pd.DataFrame(tweets)
df.to_csv('twitter_data.csv', index=False)
```

## Preparing Different Data Modalities

### 1. Text-Only Data

#### Format your CSV:
```csv
text,source,date
"This movie was absolutely terrible",reddit,2024-01-15
"I love waiting in traffic for 2 hours",twitter,2024-01-15
```

#### Prediction script:
```python
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import torch
from utils import load_config, get_device
from feature_extraction import MultimodalFeatureExtractor
from model import build_model

def predict_text_only(text: str, model_path: str):
    config = load_config('config/config.yaml')
    device = get_device(config)
    
    # Load model
    model = build_model(config, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Feature extractor
    feature_extractor = MultimodalFeatureExtractor(config, device)
    
    # For text-only, use dummy video
    dummy_video_path = "dummy.mp4"  # Will return zero tensor
    
    with torch.no_grad():
        video_feat, text_feat = feature_extractor.extract_video_text_features(
            dummy_video_path, text, 0.0, None
        )
        
        video_feat = video_feat.unsqueeze(0)
        text_feat = text_feat.unsqueeze(0)
        
        outputs = model(video_feat, text_feat)
        probs = torch.softmax(outputs, dim=1)
        
        prediction = outputs.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
    
    return {
        'text': text,
        'prediction': 'SARCASTIC' if prediction else 'NOT SARCASTIC',
        'confidence': confidence
    }

# Usage
result = predict_text_only("Your text here", "models/best_model.pth")
print(result)
```

### 2. Video-Only Data

#### From file:
```python
def predict_video_only(video_path: str, model_path: str):
    """Predict sarcasm from video without text"""
    
    config = load_config('config/config.yaml')
    device = get_device(config)
    
    model = build_model(config, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    feature_extractor = MultimodalFeatureExtractor(config, device)
    
    # Empty text
    empty_text = "[NO TEXT]"
    
    with torch.no_grad():
        video_feat, text_feat = feature_extractor.extract_video_text_features(
            video_path, empty_text, 0.0, None
        )
        
        video_feat = video_feat.unsqueeze(0)
        text_feat = text_feat.unsqueeze(0)
        
        outputs = model(video_feat, text_feat)
        probs = torch.softmax(outputs, dim=1)
        
        prediction = outputs.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
    
    return {
        'video': video_path,
        'prediction': 'SARCASTIC' if prediction else 'NOT SARCASTIC',
        'confidence': confidence
    }
```

### 3. Image-Only Data

#### Convert images to video frames:
```python
import cv2
import numpy as np
import tempfile
import os

def images_to_video(image_paths: list, output_video_path: str, fps: int = 1):
    """Convert sequence of images to video"""
    
    # Get first image to determine size
    first_image = cv2.imread(image_paths[0])
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write images to video
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (width, height))
        out.write(frame)
    
    out.release()

def predict_from_images(image_paths: list, text: str, model_path: str):
    """Predict from images (converted to video)"""
    
    # Create temporary video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        video_path = tmp.name
    
    try:
        # Convert images to video
        images_to_video(image_paths, video_path)
        
        # Use existing prediction function
        result = predict_sample(text, video_path, model_path)
        
        return result
    finally:
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)

# Usage
image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
result = predict_from_images(image_list, "Optional caption", "models/best_model.pth")
```

### 4. Batch Processing Social Media Data

```python
import pandas as pd
from tqdm import tqdm

def process_social_media_batch(csv_file: str, model_path: str, output_file: str):
    """Process entire social media dataset"""
    
    # Load data
    df = pd.read_csv(csv_file)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Get text (required)
            text = row.get('text', '')
            
            # Predict
            result = predict_text_only(text, model_path)
            
            result['source_id'] = idx
            result['source_platform'] = row.get('platform', 'unknown')
            result['original_text'] = text
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    
    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return result_df

# Usage
results = process_social_media_batch(
    'social_media_data.csv',
    'models/best_model.pth',
    'sarcasm_predictions.csv'
)
```

## Generating Explanations for Users

### Simple Explanation Template

```python
def generate_user_explanation(prediction: dict) -> str:
    """Generate user-friendly explanation"""
    
    text = prediction['text']
    is_sarcastic = prediction['prediction'] == 'SARCASTIC'
    confidence = prediction['confidence']
    
    explanation = f"""
    {'🎭 SARCASM DETECTED' if is_sarcastic else '✓ GENUINE STATEMENT'}
    
    Confidence: {confidence:.1%}
    
    Analysis:
    • Statement Text: "{text}"
    
    Why this is {'sarcastic' if is_sarcastic else 'genuine'}:
    """
    
    if is_sarcastic:
        explanation += """
    1. Language pattern suggests ironic intent
    2. Statement may contain contradictory or exaggerated elements
    3. Context indicates mocking or non-literal meaning
    """
    else:
        explanation += """
    1. Statement is straightforward and direct
    2. No contradictory patterns detected
    3. Sincere intent expressed
    """
    
    return explanation
```

## Analytics Dashboard

### Sample Analytics Script

```python
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def analyze_sarcasm_trends(predictions_csv: str):
    """Analyze sarcasm trends in social media"""
    
    df = pd.read_csv(predictions_csv)
    
    # Overall statistics
    total = len(df)
    sarcastic = (df['prediction'] == 'SARCASTIC').sum()
    genuine = total - sarcastic
    
    print(f"Total analyzed: {total}")
    print(f"Sarcastic: {sarcastic} ({sarcastic/total*100:.1f}%)")
    print(f"Genuine: {genuine} ({genuine/total*100:.1f}%)")
    
    # By platform
    if 'source_platform' in df.columns:
        platform_sarcasm = df.groupby('source_platform')['prediction'].apply(
            lambda x: (x == 'SARCASTIC').sum()
        )
        print(f"\nSarcasm by platform:\n{platform_sarcasm}")
    
    # Confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pie chart
    axes[0].pie([sarcastic, genuine], labels=['Sarcastic', 'Genuine'], autopct='%1.1f%%')
    axes[0].set_title('Sarcasm Distribution')
    
    # Histogram of confidence
    axes[1].hist(df['confidence'], bins=20)
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig('sarcasm_analysis.png')
    plt.show()

# Usage
analyze_sarcasm_trends('sarcasm_predictions.csv')
```

## Best Practices

1. **Data Privacy**: Anonymize personal information before processing
2. **Content Moderation**: Filter offensive content if needed
3. **Batch Processing**: Process large datasets in batches to avoid memory issues
4. **Result Validation**: Manually verify predictions for accuracy assessment
5. **User Feedback**: Collect feedback to improve model over time
6. **Ethical Use**: Ensure compliance with platform terms of service
7. **Bias Detection**: Monitor predictions for demographic biases

## API Integration Example

```python
from flask import Flask, request, jsonify
import torch
from utils import load_config, get_device
from model import build_model
from feature_extraction import MultimodalFeatureExtractor

app = Flask(__name__)

# Load model once at startup
config = load_config('config/config.yaml')
device = get_device(config)
model = build_model(config, device)
feature_extractor = MultimodalFeatureExtractor(config, device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    video_url = data.get('video_url')
    
    # Make prediction
    result = predict_text_only(text, model)
    
    # Generate explanation
    explanation = generate_user_explanation(result)
    
    return jsonify({
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

## Resources

- [PRAW Documentation](https://praw.readthedocs.io/)
- [Tweepy Documentation](https://docs.tweepy.org/)
- [Reddit API](https://www.reddit.com/dev/api)
- [Twitter API](https://developer.twitter.com/en/docs/twitter-api)

---

For more information, see README.md and QUICKSTART.md

