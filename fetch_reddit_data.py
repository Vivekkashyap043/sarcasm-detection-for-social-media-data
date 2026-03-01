"""
Script to fetch Reddit posts and comments using PRAW
"""
import praw
import requests
import os
from datetime import datetime
import json

# Reddit API credentials (you need to set these up)
# Get from https://www.reddit.com/prefs/apps
CLIENT_ID = 'your_client_id'
CLIENT_SECRET = 'your_client_secret'
USER_AGENT = 'SarcasmDetector/1.0'

def setup_reddit():
    """Setup PRAW Reddit instance"""
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

def download_media(url, filename):
    """Download media from URL"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def fetch_reddit_post_and_comments(post_url, reddit, download_dir='downloads'):
    """Fetch post content, comments, and media from Reddit URL"""
    os.makedirs(download_dir, exist_ok=True)

    try:
        # Get submission from URL
        submission = reddit.submission(url=post_url)

        post_data = {
            'title': submission.title,
            'selftext': submission.selftext,
            'score': submission.score,
            'subreddit': str(submission.subreddit),
            'author': str(submission.author) if submission.author else 'DELETED',
            'created_utc': submission.created_utc,
            'url': submission.url,
            'permalink': submission.permalink,
            'media': {},
            'comments': []
        }

        # Handle media
        if hasattr(submission, 'media') and submission.media:
            if 'reddit_video' in submission.media:
                video_url = submission.media['reddit_video']['fallback_url']
                video_filename = f"{download_dir}/reddit_video_{submission.id}.mp4"
                if download_media(video_url, video_filename):
                    post_data['media']['video'] = video_filename
            elif submission.url.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_filename = f"{download_dir}/image_{submission.id}{os.path.splitext(submission.url)[1]}"
                if download_media(submission.url, image_filename):
                    post_data['media']['image'] = image_filename

        # Fetch comments
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            comment_data = {
                'author': str(comment.author) if comment.author else 'DELETED',
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'permalink': comment.permalink
            }
            post_data['comments'].append(comment_data)

        return post_data

    except Exception as e:
        print(f"Error fetching {post_url}: {e}")
        return None

def main():
    # Your provided data
    reddit_data = [
        {
            "title": "How I learned Python",
            "text": "I spent the last year learning Python...",
            "subreddit": "PythonLearning",
            "score": 122,
            "url": "https://www.reddit.com/r/PythonLearning/comments/1obwkbb/how_i_learned_python/"
        },
        # ... add other posts
    ]

    reddit = setup_reddit()

    all_data = []
    for post in reddit_data:
        print(f"Fetching: {post['title']}")
        data = fetch_reddit_post_and_comments(post['url'], reddit)
        if data:
            all_data.append(data)

    # Save to JSON
    with open('fetched_reddit_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Fetched {len(all_data)} posts with comments and media")

if __name__ == "__main__":
    main()