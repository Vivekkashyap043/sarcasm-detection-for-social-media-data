"""
Fetch Reddit data using Reddit's JSON API (no auth required)
"""
import requests
import json
import time
import os
import re
from urllib.parse import urlparse

# Import model components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_config, get_device
from feature_extraction import MultimodalFeatureExtractor
from model import build_model
import torch

def fetch_reddit_posts_json(subreddit, limit=5):
    """Fetch posts from subreddit using Reddit's JSON API, searching for 'India vs Pakistan'"""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    headers = {
        'User-Agent': 'SarcasmDetector/1.0 (by /u/test_user)'
    }
    params = {
        'q': 'India vs Pakistan',
        'restrict_sr': 1,
        'sort': 'new',
        'limit': limit
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data['data']['children']
        else:
            print(f"API returned status {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching from Reddit: {e}")
        return []

def fetch_post_comments_json(subreddit, post_id, limit=10):
    """Fetch comments for a post using Reddit's JSON API"""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    headers = {
        'User-Agent': 'SarcasmDetector/1.0 (by /u/test_user)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and 'data' in data[1]:
                comments = data[1]['data']['children']
                return [c['data'] for c in comments if c['kind'] == 't1'][:limit]
        return []
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

def download_reddit_video(video_url, filename):
    """Download video from Reddit's CDN in correct MP4 format (fallback to best available)"""
    try:
        # Try fallback_url first (should be a direct MP4)
        response = requests.get(video_url, timeout=60)
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('video'):
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
        # If fallback_url fails, try DASH qualities
        base_url = video_url.replace('DASH_', '').rsplit('/', 1)[0]
        qualities = ['720', '480', '360', '240']
        for quality in qualities:
            video_file = f"{base_url}/DASH_{quality}.mp4"
            response = requests.get(video_file, timeout=60)
            if response.status_code == 200 and response.headers.get('content-type', '').startswith('video'):
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return True
        return False
    except Exception as e:
        print(f"Failed to download video: {e}")
        return False

def download_media(url, filename):
    """Download image or external media"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download media: {e}")
    return False

def extract_media_info(post_data):
    """Extract media URLs from post data"""
    media_info = {
        'video_url': None,
        'image_url': None,
        'is_video': False,
        'is_image': False
    }

    # Check for Reddit video
    if 'media' in post_data and post_data['media'] and 'reddit_video' in post_data['media']:
        media_info['video_url'] = post_data['media']['reddit_video']['fallback_url']
        media_info['is_video'] = True

    # Check for images
    elif 'url' in post_data:
        url = post_data['url']
        if url.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            media_info['image_url'] = url
            media_info['is_image'] = True

    return media_info

def test_sarcasm(text, video_path=None, image_path=None, model=None, feature_extractor=None, device=None):
    """Test sarcasm on text + media"""
    if not model or not text.strip():
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

def process_reddit_data(subreddits=['Python', 'programming', 'learnpython'], posts_per_sub=3):
    """Main function to fetch Reddit data and test sarcasm for 'India vs Pakistan'"""

    # Create directories
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

    all_results = []
    all_explanations = []
    # Import SarcasmExplainer
    from explainability import SarcasmExplainer
    explainer = SarcasmExplainer(model, feature_extractor, config, device)

    # Use cricket and sports subreddits for this topic
    subreddits = ['Cricket', 'sports', 'india', 'pakistan', 'worldnews']
    posts_per_sub = 5
    for subreddit in subreddits:
        print(f"\n=== Processing r/{subreddit} for 'India vs Pakistan' ===")

        posts = fetch_reddit_posts_json(subreddit, limit=posts_per_sub)

        for post in posts:
            post_data = post['data']
            post_id = post_data['id']
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')
            score = post_data.get('score', 0)
            post_url = post_data.get('permalink', '')

            print(f"\nProcessing post: {title[:50]}...")

            # Combine title and text
            full_text = f"{title} {selftext}".strip()

            # Extract media info
            media_info = extract_media_info(post_data)

            # Download media
            video_path = None
            image_path = None

            if media_info['is_video'] and media_info['video_url']:
                video_filename = f"downloads/videos/{post_id}.mp4"
                if download_reddit_video(media_info['video_url'], video_filename):
                    video_path = video_filename
                    print("Downloaded video")
                else:
                    print("Failed to download video")

            elif media_info['is_image'] and media_info['image_url']:
                ext = os.path.splitext(media_info['image_url'])[1] or '.jpg'
                image_filename = f"downloads/images/{post_id}{ext}"
                if download_media(media_info['image_url'], image_filename):
                    image_path = image_filename
                    print("Downloaded image")

            # Test post sarcasm and generate explanation
            if full_text:
                result = test_sarcasm(full_text, video_path, image_path, model, feature_extractor, device)
                if result:
                    post_result = {
                        'type': 'post',
                        'subreddit': subreddit,
                        'title': title,
                        'text': full_text,
                        'score': score,
                        'url': f"https://reddit.com{post_url}",
                        'media': {
                            'has_video': video_path is not None,
                            'has_image': image_path is not None,
                            'video_path': video_path,
                            'image_path': image_path
                        },
                        'sarcasm_result': result
                    }
                    all_results.append(post_result)
                    print(f"Post result: {result['prediction']} ({result['confidence']:.3f})")

                    # Generate explainability output
                    try:
                        explanation = explainer.explain_prediction(
                            text=full_text,
                            video_path=video_path if video_path else '',
                            ground_truth_label=None
                        )
                        explanation['source_type'] = 'post'
                        explanation['subreddit'] = subreddit
                        explanation['title'] = title
                        explanation['url'] = f"https://reddit.com{post_url}"
                        all_explanations.append(explanation)
                    except Exception as e:
                        print(f"Explainability error (post): {e}")

            # Fetch and test comments, generate explanations
            comments = fetch_post_comments_json(subreddit, post_id, limit=5)

            for comment in comments:
                comment_text = comment.get('body', '')
                if comment_text and len(comment_text) > 10:  # Skip short comments
                    result = test_sarcasm(comment_text, video_path, image_path, model, feature_extractor, device)
                    if result:
                        comment_result = {
                            'type': 'comment',
                            'subreddit': subreddit,
                            'post_title': title,
                            'text': comment_text,
                            'score': comment.get('score', 0),
                            'sarcasm_result': result
                        }
                        all_results.append(comment_result)
                        print(f"Comment result: {result['prediction']} ({result['confidence']:.3f})")

                        # Generate explainability output
                        try:
                            explanation = explainer.explain_prediction(
                                text=comment_text,
                                video_path=video_path if video_path else '',
                                ground_truth_label=None
                            )
                            explanation['source_type'] = 'comment'
                            explanation['subreddit'] = subreddit
                            explanation['post_title'] = title
                            all_explanations.append(explanation)
                        except Exception as e:
                            print(f"Explainability error (comment): {e}")

            # Rate limiting
            time.sleep(2)

    # Save results
    with open('results/reddit_sarcasm_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save explainability outputs
    with open('results/reddit_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False)

    # Generate summary TXT
    summary_file = 'results/reddit_explanations_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REDDIT SARCASM EXPLAINABILITY SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for i, exp in enumerate(all_explanations):
            f.write(f"Sample {i + 1}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Source: {exp.get('source_type', '')}\n")
            f.write(f"Subreddit: {exp.get('subreddit', '')}\n")
            f.write(f"Title: {exp.get('title', exp.get('post_title', ''))[:100]}\n")
            f.write(f"Text: {exp['text'][:100]}...\n\n")
            f.write(f"Prediction: {exp['prediction_label_name']} (confidence: {exp['predicted_probability']:.4f})\n")
            if 'ground_truth_label_name' in exp:
                f.write(f"Ground Truth: {exp['ground_truth_label_name']}\n")
                f.write(f"Correct: {exp['is_correct']}\n")
            f.write("\n")
            # Print top contributing words
            if 'text_explanation' in exp:
                text_exp = exp['text_explanation']
                if 'feature_importance' in text_exp:
                    f.write("Top Contributing Words:\n")
                    sorted_features = sorted(
                        text_exp['feature_importance'].items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    for word, importance in sorted_features[:5]:
                        f.write(f"  {word:20} : {importance:8.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print("\n=== Summary ===")
    sarcastic = sum(1 for r in all_results if r['sarcasm_result']['prediction'] == 'SARCASTIC')
    print(f"Total items processed: {len(all_results)}")
    print(f"Sarcastic content found: {sarcastic} ({sarcastic/len(all_results)*100:.1f}%)")
    print("Results saved to results/reddit_sarcasm_results.json")
    print("Explainability outputs saved to results/reddit_explanations.json and summary TXT.")

if __name__ == "__main__":
    process_reddit_data()