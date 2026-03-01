"""
Test script with mock Reddit data including videos
"""
import os
import json
import tempfile
import requests
from pathlib import Path

# Import model components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_config, get_device
from feature_extraction import MultimodalFeatureExtractor
from model import build_model
import torch

def download_sample_video(url, filename):
    """Download a sample video for testing"""
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download video: {e}")
    return False

def test_sarcasm_with_mock_data():
    """Test sarcasm detection with mock Reddit data including videos"""

    # Load model
    config = load_config("config/config.yaml")
    device = get_device(config)

    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("best_model") and f.endswith(".pth")]
    model_files.sort(reverse=True)
    model_path = os.path.join(models_dir, model_files[0])

    model = build_model(config, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    feature_extractor = MultimodalFeatureExtractor(config, device)

    # Mock Reddit data with videos and comments
    mock_data = [
        {
            "type": "post",
            "title": "Finally got my Python code working!",
            "text": "After hours of debugging, my script finally runs without errors. Python is such a great language!",
            "subreddit": "Python",
            "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",  # Sample video
            "comments": [
                {"text": "Congrats! What was the issue?", "author": "user1"},
                {"text": "Python is amazing, isn't it? So clean and readable.", "author": "user2"},
                {"text": "Oh sure, because nothing says 'great language' like spending hours debugging basic syntax errors. 🙄", "author": "sarcastic_user"}
            ]
        },
        {
            "type": "post",
            "title": "I love how Python makes everything so easy",
            "text": "Just wrote a 200-line script in 10 minutes. Python is the best!",
            "subreddit": "learnpython",
            "video_url": None,
            "comments": [
                {"text": "That's awesome! What did you build?", "author": "curious_dev"},
                {"text": "200 lines in 10 minutes? Wow, you must be a genius. Or maybe you copied it from Stack Overflow. 😏", "author": "skeptic_dev"}
            ]
        }
    ]

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for post in mock_data:
            print(f"\n--- Testing Post: {post['title']} ---")

            # Download video if available
            video_path = None
            if post.get('video_url'):
                video_filename = os.path.join(temp_dir, f"video_{len(results)}.mp4")
                if download_sample_video(post['video_url'], video_filename):
                    video_path = video_filename
                    print("Downloaded sample video")
                else:
                    print("Failed to download video")

            # Test post
            post_text = f"{post['title']} {post['text']}"
            with torch.no_grad():
                video_feat, text_feat = feature_extractor.extract_video_text_features(
                    video_path, post_text
                )
                video_feat = video_feat.unsqueeze(0)
                text_feat = text_feat.unsqueeze(0)

                outputs = model(video_feat, text_feat)
                probs = torch.softmax(outputs, dim=1)
                pred_label = outputs.argmax(dim=1).item()
                confidence = probs[0].cpu().numpy()

                post_result = {
                    "type": "post",
                    "title": post['title'],
                    "text": post_text,
                    "prediction": "SARCASTIC" if pred_label else "NOT SARCASTIC",
                    "confidence": float(probs[0, pred_label].item()),
                    "probabilities": {
                        "not_sarcastic": float(confidence[0]),
                        "sarcastic": float(confidence[1])
                    },
                    "has_video": video_path is not None
                }
                results.append(post_result)
                print(f"Post: {post_result['prediction']} ({post_result['confidence']:.3f})")

            # Test comments
            for i, comment in enumerate(post['comments']):
                comment_text = comment['text']
                with torch.no_grad():
                    video_feat, text_feat = feature_extractor.extract_video_text_features(
                        video_path, comment_text  # Use post's video for comment
                    )
                    video_feat = video_feat.unsqueeze(0)
                    text_feat = text_feat.unsqueeze(0)

                    outputs = model(video_feat, text_feat)
                    probs = torch.softmax(outputs, dim=1)
                    pred_label = outputs.argmax(dim=1).item()
                    confidence = probs[0].cpu().numpy()

                    comment_result = {
                        "type": "comment",
                        "post_title": post['title'],
                        "text": comment_text,
                        "prediction": "SARCASTIC" if pred_label else "NOT SARCASTIC",
                        "confidence": float(probs[0, pred_label].item()),
                        "probabilities": {
                            "not_sarcastic": float(confidence[0]),
                            "sarcastic": float(confidence[1])
                        },
                        "has_video": video_path is not None
                    }
                    results.append(comment_result)
                    print(f"Comment {i+1}: {comment_result['prediction']} ({comment_result['confidence']:.3f}) - {comment_text[:50]}...")

    # Save results
    with open('results/mock_reddit_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    sarcastic = sum(1 for r in results if r['prediction'] == 'SARCASTIC')
    print("\n=== Summary ===")
    print(f"Total items tested: {len(results)}")
    print(f"Sarcastic content: {sarcastic} ({sarcastic/len(results)*100:.1f}%)")
    print("Results saved to results/mock_reddit_test_results.json")

if __name__ == "__main__":
    test_sarcasm_with_mock_data()