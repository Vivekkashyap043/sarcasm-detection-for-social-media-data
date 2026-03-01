import requests
import json
import os
import time
from urllib.parse import urlparse

# Model imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import load_config, get_device
from feature_extraction import MultimodalFeatureExtractor
from model import build_model
import torch

# -----------------------------
# 🔐 RapidAPI Credentials
# -----------------------------
RAPIDAPI_KEY = "f92d7c07e1mshfeae10cff4259b6p1383a3jsna95268104342"
RAPIDAPI_HOST = "reddit34.p.rapidapi.com"


def search_reddit(query, max_results=10):
    url = "https://reddit34.p.rapidapi.com/getSearchPosts"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    params = {"query": query, "limit": max_results}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print("❌ API Error:", response.status_code)
        print(response.text)
        return []
    data = response.json()
    if not data.get("success"):
        print("❌ API returned success=False")
        return []
    posts = data["data"]["posts"]
    structured_posts = []
    for post in posts:
        post_data = post["data"]
        structured_posts.append({
            "title": post_data.get("title"),
            "text": post_data.get("selftext"),
            "subreddit": post_data.get("subreddit"),
            "score": post_data.get("score"),
            "url": post_data.get("url")
        })
    return structured_posts

def download_media(url, filename):
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download media: {e}")
    return False

def is_video_url(url):
    if not url:
        return False
    video_exts = ['.mp4', '.webm', '.mov', '.avi']
    return any(url.lower().endswith(ext) for ext in video_exts) or 'v.redd.it' in url

def is_image_url(url):
    if not url:
        return False
    image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    return any(url.lower().endswith(ext) for ext in image_exts)

def test_sarcasm(text, video_path=None, image_path=None, model=None, feature_extractor=None, device=None):
    if not model or not text or not text.strip():
        return None
    try:
        with torch.no_grad():
            video_feat, text_feat = feature_extractor.extract_video_text_features(
                video_path, text, image_path=image_path
            )
            video_feat = video_feat.unsqueeze(0)
            text_feat = text_feat.unsqueeze(0)
            outputs = model(video_feat, text_feat)
            probs = torch.softmax(outputs, dim=1)
            pred_label = outputs.argmax(dim=1).item()
            return {
                'prediction': 'SARCASTIC' if pred_label else 'NOT SARCASTIC',
                'confidence': float(probs[0, pred_label].item()),
                'probabilities': {
                    'not_sarcastic': float(probs[0, 0].item()),
                    'sarcastic': float(probs[0, 1].item())
                }
            }
    except Exception as e:
        print(f"Error testing sarcasm: {e}")
        return None

def fetch_and_test_all(query="python", max_results=5):

    os.makedirs('downloads/videos', exist_ok=True)
    os.makedirs('downloads/images', exist_ok=True)
    os.makedirs('results', exist_ok=True)
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
    # Fetch posts for 'India vs Pakistan'
    posts = search_reddit("India vs Pakistan", max_results)
    all_results = []
    for i, post in enumerate(posts):
        print(f"\n--- Post {i+1} ---")
        print("Title:", post["title"])
        print("Subreddit:", post["subreddit"])
        print("Score:", post["score"])
        print("Text:", post["text"] if post["text"] else "No body text")
        print("URL:", post["url"])
        video_path = None
        image_path = None
        url = post["url"]
        # Download video in correct format (MP4)
        if is_video_url(url):
            video_filename = f"downloads/videos/post_{i+1}.mp4"
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200 and resp.headers.get('content-type', '').startswith('video'):
                    with open(video_filename, 'wb') as f:
                        f.write(resp.content)
                    video_path = video_filename
                    print("Downloaded video:", video_filename)
                else:
                    print("Failed to download video or not a video file.")
            except Exception as e:
                print(f"Failed to download video: {e}")
        elif is_image_url(url):
            ext = os.path.splitext(url)[1] or '.jpg'
            image_filename = f"downloads/images/post_{i+1}{ext}"
            if download_media(url, image_filename):
                image_path = image_filename
                print("Downloaded image:", image_filename)
        # Run model
        result = test_sarcasm(post["text"] or post["title"], video_path, image_path, model, feature_extractor, device)
        if result:
            print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
            all_results.append({
                'title': post['title'],
                'text': post['text'],
                'subreddit': post['subreddit'],
                'score': post['score'],
                'url': post['url'],
                'media': {'video': video_path, 'image': image_path},
                'sarcasm_result': result
            })
        time.sleep(1)
    # Save results
    with open('results/rapidapi_reddit_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\n✅ Data and results saved to 'results/rapidapi_reddit_results.json'")

if __name__ == "__main__":
    fetch_and_test_all(query="python", max_results=5)
