"""
Alternative: Fetch Reddit data using Pushshift API (no authentication needed)
"""
import requests
import json
from datetime import datetime
import time

def fetch_reddit_posts_pushshift(subreddit, limit=10, after=None):
    """Fetch posts from subreddit using Pushshift API"""
    base_url = "https://api.pushshift.io/reddit/search/submission/"

    params = {
        'subreddit': subreddit,
        'size': limit,
        'sort': 'desc',
        'sort_type': 'created_utc'
    }

    if after:
        params['after'] = after

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data['data']
    except Exception as e:
        print(f"Error fetching from Pushshift: {e}")

    return []

def fetch_reddit_comments_pushshift(subreddit, link_id, limit=50):
    """Fetch comments for a post using Pushshift API"""
    base_url = "https://api.pushshift.io/reddit/search/comment/"

    params = {
        'subreddit': subreddit,
        'link_id': link_id,  # t3_xxxxxx format
        'size': limit,
        'sort': 'desc',
        'sort_type': 'created_utc'
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data['data']
    except Exception as e:
        print(f"Error fetching comments from Pushshift: {e}")

    return []

def download_media(url, filename):
    """Download media if it's an image/video"""
    if not url or not url.startswith('http'):
        return False

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def process_reddit_data(subreddit='Python', num_posts=5):
    """Fetch posts and comments from subreddit"""
    posts = fetch_reddit_posts_pushshift(subreddit, limit=num_posts)

    all_data = []

    for post in posts:
        post_data = {
            'title': post.get('title', ''),
            'selftext': post.get('selftext', ''),
            'score': post.get('score', 0),
            'subreddit': post.get('subreddit', ''),
            'author': post.get('author', 'DELETED'),
            'created_utc': post.get('created_utc', 0),
            'url': post.get('url', ''),
            'permalink': post.get('permalink', ''),
            'media': {},
            'comments': []
        }

        # Handle media
        if post.get('url', '').endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Download image
            ext = post['url'].split('.')[-1]
            filename = f"downloads/image_{post['id']}.{ext}"
            if download_media(post['url'], filename):
                post_data['media']['image'] = filename

        # Fetch comments
        link_id = f"t3_{post['id']}"
        comments = fetch_reddit_comments_pushshift(subreddit, link_id, limit=20)

        for comment in comments:
            comment_data = {
                'author': comment.get('author', 'DELETED'),
                'body': comment.get('body', ''),
                'score': comment.get('score', 0),
                'created_utc': comment.get('created_utc', 0)
            }
            post_data['comments'].append(comment_data)

        all_data.append(post_data)

        # Rate limiting
        time.sleep(1)

    return all_data

def main():
    # Example: Fetch from r/Python
    data = process_reddit_data('Python', num_posts=3)

    # Save to JSON
    with open('pushshift_reddit_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Fetched {len(data)} posts with comments")

if __name__ == "__main__":
    main()