"""
Alternative: Simple web scraping for Reddit posts (basic text extraction)
Note: This is for educational purposes only. Check Reddit's TOS.
"""
import requests
from bs4 import BeautifulSoup
import json
import re
import os

def scrape_reddit_post(url):
    """Scrape basic text content from Reddit post URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title_elem = soup.find('h1', {'class': re.compile(r'.*title.*')})
        title = title_elem.get_text().strip() if title_elem else "No title"

        # Extract post text
        post_text_elem = soup.find('div', {'class': re.compile(r'.*usertext-body.*')})
        post_text = ""
        if post_text_elem:
            post_text = post_text_elem.get_text().strip()

        # Extract comments (basic)
        comments = []
        comment_elems = soup.find_all('div', {'class': re.compile(r'.*comment.*')})
        for comment in comment_elems[:10]:  # Limit to first 10
            comment_text = comment.get_text().strip()
            if len(comment_text) > 50:  # Filter short comments
                comments.append({
                    'body': comment_text[:500],  # Truncate
                    'author': 'Unknown'
                })

        return {
            'title': title,
            'selftext': post_text,
            'url': url,
            'comments': comments,
            'media': {}  # Can't easily extract videos via scraping
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    # Your provided URLs
    urls = [
        "https://www.reddit.com/r/PythonLearning/comments/1obwkbb/how_i_learned_python/",
        "https://www.reddit.com/r/PythonProjects2/comments/1qvbq2y/i_quit_python_learning/",
        "https://i.redd.it/udrizysfjjcg1.png",  # Skip images
        "https://i.redd.it/e8qgxdcyltyf1.jpeg",  # Skip images
        "https://www.reddit.com/r/learnpython/comments/1qnkhrn/is_it_worth_learning_python/",
        "https://i.redd.it/n2tpwiocry4g1.png",  # Skip images
        "https://www.reddit.com/gallery/1nrrgwi",  # Gallery
        "https://www.reddit.com/r/FinancialCareers/comments/1p3pk3b/has_python_become_irrelevant/"
    ]

    scraped_data = []

    for url in urls:
        if url.startswith('https://www.reddit.com'):
            print(f"Scraping: {url}")
            data = scrape_reddit_post(url)
            if data:
                scraped_data.append(data)

    # Save
    with open('scraped_reddit_data.json', 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)

    print(f"Scraped {len(scraped_data)} posts")

if __name__ == "__main__":
    main()