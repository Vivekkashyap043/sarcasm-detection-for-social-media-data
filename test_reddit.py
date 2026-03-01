"""
Script to test sarcasm detection on Reddit data
"""
import os
import sys
import requests
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_config, get_device
from feature_extraction import MultimodalFeatureExtractor
from model import build_model
import torch

def download_image(url, temp_dir):
    """Download image from URL to temp file"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Get file extension
            ext = '.png' if url.endswith('.png') else '.jpg' if url.endswith('.jpg') else '.jpeg'
            temp_path = os.path.join(temp_dir, f"temp_image{ext}")
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            return temp_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return None

def predict_on_reddit_data():
    # Load config
    config = load_config("config/config.yaml")
    device = get_device(config)

    # Load model - use the latest best model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("best_model") and f.endswith(".pth")]
    if not model_files:
        print("No model files found!")
        return
    # Sort by date/time in filename
    model_files.sort(reverse=True)
    model_path = os.path.join(models_dir, model_files[0])
    print(f"Using model: {model_path}")

    # Build model
    model = build_model(config, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Feature extractor
    feature_extractor = MultimodalFeatureExtractor(config, device)

    # Reddit data
    reddit_data = [
        {
            "title": "How I learned Python",
            "text": "I spent the last year learning Python and producing an animated Discord bot with thermal monitoring, persistent learning, deterministic particle effects, and a lot more. It's a lot of work but I was able to learn an insane amount quickly. I was wondering if anyone wanted help getting going on Python?\n\nIm a teacher professionally and think the way I learned was really accelerated. I was going to offer it to others if anyone needs help. \n\nLet me know! ",
            "subreddit": "PythonLearning",
            "score": 122,
            "url": "https://www.reddit.com/r/PythonLearning/comments/1obwkbb/how_i_learned_python/"
        },
        {
            "title": "I QUIT PYTHON LEARNING",
            "text": "I\u2019ve been learning Python using ChatGPT, starting from zero. I actually learned a lot more than I expected \u2014 variables, loops, lists, tuples, dicts, functions, and basic problem-solving. The interactive part helped a lot: asking \u201cwhy\u201d, testing myself, fixing logic, etc.\n\nI\u2019d say I reached an early\u2013intermediate level and genuinely understood what I was doing.\n\nThen I hit classes.\n\nThat topic completely killed my momentum. No matter how many explanations or examples I saw, the class/object/self/init stuff just felt abstract and unnecessary compared to everything before it. I got frustrated, motivation dropped, and I decided to stop instead of forcing it.\n\nAt this point, I\u2019m honestly thinking of quitting this programming language altogether. Maybe it\u2019s not for me\n\nJust sharing in case anyone else is learning Python the same way and hits the same wall. You\u2019re not alone.\n\n\ud83d\ude43\n\nGoodbye ",
            "subreddit": "PythonProjects2",
            "score": 77,
            "url": "https://www.reddit.com/r/PythonProjects2/comments/1qvbq2y/i_quit_python_learning/"
        },
        {
            "title": "I fucking hate python",
            "text": "",
            "subreddit": "programmingmemes",
            "score": 499,
            "url": "https://i.redd.it/udrizysfjjcg1.png"
        },
        {
            "title": "List of functions in Python",
            "text": "Hello,\n\nIs a list with the same visual appearance as in the image also available in Python?",
            "subreddit": "PythonLearning",
            "score": 731,
            "url": "https://i.redd.it/e8qgxdcyltyf1.jpeg"
        },
        {
            "title": "Is it worth learning Python?",
            "text": "Is it too late to learn Python? What can I do with it, especially if I want to develop micro SaaS applications, web applications, work with data, and build artificial intelligence solutions?",
            "subreddit": "learnpython",
            "score": 0,
            "url": "https://www.reddit.com/r/learnpython/comments/1qnkhrn/is_it_worth_learning_python/"
        },
        {
            "title": "Python is just different guys..",
            "text": "",
            "subreddit": "devhumormemes",
            "score": 2163,
            "url": "https://i.redd.it/n2tpwiocry4g1.png"
        },
        {
            "title": "My third python code",
            "text": "",
            "subreddit": "PythonLearning",
            "score": 77,
            "url": "https://www.reddit.com/gallery/1nrrgwi"
        },
        {
            "title": "Has Python become irrelevant?",
            "text": "I went to Morgan Stanley for interview for summer internship, where 2 other candidates were talking about the irrelevance of Python, how his manager uses AI for python even though he knows to code, and how powerbi is a more powerful tool to learn.\n\nAny comments or insights on this?",
            "subreddit": "FinancialCareers",
            "score": 143,
            "url": "https://www.reddit.com/r/FinancialCareers/comments/1p3pk3b/has_python_become_irrelevant/"
        }
    ]

    # Temp dir for images
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, post in enumerate(reddit_data):
            print(f"\n--- Post {i+1}: {post['title']} ---")
            text = post['text'] or post['title']  # Use title if text empty
            url = post['url']

            # Check if URL is an image
            image_path = None
            if url.startswith("https://i.redd.it/"):
                image_path = download_image(url, temp_dir)
                if image_path:
                    print(f"Downloaded image: {image_path}")
                else:
                    print("Failed to download image")

            # Predict
            with torch.no_grad():
                video_feat, text_feat = feature_extractor.extract_video_text_features(
                    video_path=None,
                    text=text,
                    image_path=image_path
                )

                # Add batch dim
                video_feat = video_feat.unsqueeze(0)
                text_feat = text_feat.unsqueeze(0)

                outputs = model(video_feat, text_feat)
                probs = torch.softmax(outputs, dim=1)

                pred_label = outputs.argmax(dim=1).item()
                pred_prob = probs[0, pred_label].item()
                confidence = probs[0].cpu().numpy()

            print(f"Text: {text[:100]}...")
            print(f"Prediction: {'SARCASTIC' if pred_label else 'NOT SARCASTIC'}")
            print(f"Confidence: {pred_prob:.3f}")
            print(f"Probabilities: Not Sarcastic: {confidence[0]:.3f}, Sarcastic: {confidence[1]:.3f}")

if __name__ == "__main__":
    predict_on_reddit_data()