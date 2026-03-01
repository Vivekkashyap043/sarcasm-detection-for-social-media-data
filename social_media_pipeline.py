"""Fetch Reddit content, download media, and run multimodal sarcasm inference."""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from src.inference import MultimodalSarcasmPredictor
from src.utils import setup_logging

logger = setup_logging(level="INFO")

USER_AGENT = "multimodal-sarcasm-detector/1.0"


def is_video_url(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return any(lowered.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv']) or 'v.redd.it' in lowered


def is_image_url(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return any(lowered.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])


def _safe_filename(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or fallback
    name = name.split('?')[0]
    return ''.join(ch if ch.isalnum() or ch in ['.', '_', '-'] else '_' for ch in name)


def _resolve_reddit_video_url(url: str) -> str:
    if 'v.redd.it' not in url:
        return url

    clean = url.split('?')[0].rstrip('/')
    parts = clean.split('/')
    if len(parts) < 4:
        return url

    video_id = parts[3]
    for quality in ['DASH_720.mp4', 'DASH_480.mp4', 'DASH_360.mp4', 'DASH_240.mp4']:
        candidate = f"https://v.redd.it/{video_id}/{quality}"
        try:
            resp = requests.head(
                candidate,
                timeout=10,
                allow_redirects=True,
                headers={"User-Agent": USER_AGENT}
            )
            if resp.status_code == 200:
                return candidate
        except Exception:
            continue

    return url


def download_media(url: str, output_dir: str, max_size_mb: int = 80) -> Optional[str]:
    if not url or not url.startswith('http'):
        return None

    os.makedirs(output_dir, exist_ok=True)
    final_url = _resolve_reddit_video_url(url)

    try:
        with requests.get(final_url, stream=True, timeout=90, headers={"User-Agent": USER_AGENT}) as response:
            if response.status_code != 200:
                return None

            content_type = (response.headers.get('content-type') or '').lower()
            if content_type and not (content_type.startswith('video/') or content_type.startswith('image/')):
                logger.warning(f"Skipping non-media response for URL: {final_url} (content-type: {content_type})")
                return None

            file_name = _safe_filename(final_url, fallback="media.bin")
            root, ext = os.path.splitext(file_name)
            if not ext:
                if content_type.startswith('video/'):
                    ext = '.mp4'
                elif content_type.startswith('image/'):
                    ext = '.jpg'
                file_name = f"{root}{ext}" if ext else file_name

            output_path = os.path.join(output_dir, file_name)

            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                logger.warning(f"Skipping oversized media: {final_url}")
                return None

            downloaded = 0
            with open(output_path, 'wb') as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_size_mb * 1024 * 1024:
                        logger.warning(f"Aborting oversized media: {final_url}")
                        handle.close()
                        os.remove(output_path)
                        return None
                    handle.write(chunk)

        return output_path
    except Exception as error:
        logger.warning(f"Media download failed ({url}): {error}")
        return None


def fetch_subreddit_posts(subreddit: str, limit: int) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    params = {"limit": min(max(limit, 1), 100)}
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        children = payload.get('data', {}).get('children', [])
        return [child.get('data', {}) for child in children]
    except Exception as error:
        logger.warning(f"Failed to fetch subreddit {subreddit}: {error}")
        return []


def fetch_keyword_posts(keyword: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch Reddit posts globally by keyword."""
    url = "https://www.reddit.com/search.json"
    params = {
        "q": keyword,
        "limit": min(max(limit, 1), 100),
        "sort": "new",
        "type": "link"
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        children = payload.get('data', {}).get('children', [])
        return [child.get('data', {}) for child in children]
    except Exception as error:
        logger.warning(f"Failed to fetch keyword '{keyword}': {error}")
        return []


def fetch_post_comments(permalink: str, limit: int) -> List[Dict[str, Any]]:
    if not permalink:
        return []

    url = f"https://www.reddit.com{permalink}.json"
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if len(payload) < 2:
            return []

        comments = []
        children = payload[1].get('data', {}).get('children', [])
        for child in children[:limit]:
            if child.get('kind') != 't1':
                continue
            data = child.get('data', {})
            body = data.get('body', '')
            if body and body not in ['[removed]', '[deleted]']:
                comments.append(data)
        return comments
    except Exception as error:
        logger.warning(f"Failed to fetch comments for {permalink}: {error}")
        return []


def classify_reddit_content(
    predictor: MultimodalSarcasmPredictor,
    subreddit: str,
    posts: List[Dict[str, Any]],
    comment_limit: int,
    download_root: str,
    source_keyword: Optional[str] = None,
    detailed_explanations: bool = False,
) -> List[Dict[str, Any]]:
    results = []

    for post in posts:
        post_id = post.get('id', '')
        title = post.get('title', '') or ''
        selftext = post.get('selftext', '') or ''
        permalink = post.get('permalink', '') or ''
        url = post.get('url', '') or ''
        media_info = post.get('preview', {})

        input_text = (title + "\n" + selftext).strip()

        video_path = None
        image_path = None

        post_media_dir = os.path.join(download_root, subreddit, post_id)
        if is_video_url(url):
            video_path = download_media(url, post_media_dir)
        elif is_image_url(url):
            image_path = download_media(url, post_media_dir)

        if not image_path and isinstance(media_info, dict):
            image_candidates = media_info.get('images', [])
            if image_candidates:
                source = image_candidates[0].get('source', {})
                image_url = source.get('url', '').replace('&amp;', '&')
                if image_url:
                    image_path = download_media(image_url, post_media_dir)

        if input_text or video_path or image_path:
            prediction = predictor.predict(
                text=input_text if input_text else None,
                video_path=video_path,
                image_path=image_path,
                audio_path=None,
                generate_detailed_explanation=detailed_explanations,
            )

            sarcasm_result = {
                "prediction": prediction.get("label_name", "NOT SARCASTIC"),
                "confidence": float(prediction.get("confidence", 0.0)),
                "probabilities": prediction.get("probabilities", {}),
                "explanation": prediction.get("detailed_explanation", {})
            }

            results.append({
                "type": "post",
                "subreddit": subreddit,
                "title": title,
                "text": input_text,
                "score": int(post.get('score', 0) or 0),
                "url": url,
                "media": {
                    "video": video_path,
                    "image": image_path,
                },
                "sarcasm_result": sarcasm_result,
                "keyword": source_keyword,
                "post_id": post_id,
                "permalink": permalink,
            })

            label = prediction.get("label_name", "UNKNOWN")
            confidence = prediction.get("confidence", 0.0)
            logger.info(
                f"[POST] r/{subreddit} | {title[:70]} -> {label} ({confidence:.3f})"
            )

        comments = fetch_post_comments(permalink, comment_limit)
        for comment in comments:
            body = comment.get('body', '')
            if not body.strip():
                continue

            prediction = predictor.predict(
                text=body,
                video_path=video_path,
                image_path=image_path,
                audio_path=None,
                generate_detailed_explanation=detailed_explanations,
            )

            sarcasm_result = {
                "prediction": prediction.get("label_name", "NOT SARCASTIC"),
                "confidence": float(prediction.get("confidence", 0.0)),
                "probabilities": prediction.get("probabilities", {}),
                "explanation": prediction.get("detailed_explanation", {})
            }

            results.append({
                "type": "comment",
                "subreddit": subreddit,
                "post_title": title,
                "comment_id": comment.get('id', ''),
                "text": body,
                "score": comment.get('score', 0),
                "media": {
                    "video": video_path,
                    "image": image_path,
                },
                "sarcasm_result": sarcasm_result,
                "keyword": source_keyword,
                "post_id": post_id,
            })

            label = prediction.get("label_name", "UNKNOWN")
            confidence = prediction.get("confidence", 0.0)
            logger.info(
                f"[COMMENT] r/{subreddit} | {body[:70]} -> {label} ({confidence:.3f})"
            )

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def run_pipeline(
    keywords: Optional[List[str]],
    subreddits: List[str],
    posts_per_subreddit: int,
    comments_per_post: int,
    config_path: str,
    model_path: Optional[str],
    output_path: str,
    detailed_explanations: bool,
):
    predictor = MultimodalSarcasmPredictor(config_path=config_path, model_path=model_path)

    all_results: List[Dict[str, Any]] = []
    download_root = os.path.join(Path(output_path).parent, 'downloads')

    if keywords:
        for keyword in keywords:
            logger.info(f"Processing keyword: {keyword}")
            posts = fetch_keyword_posts(keyword, posts_per_subreddit)
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for post in posts:
                sub = post.get('subreddit', 'unknown')
                grouped.setdefault(sub, []).append(post)

            for subreddit, sub_posts in grouped.items():
                sub_results = classify_reddit_content(
                    predictor=predictor,
                    subreddit=subreddit,
                    posts=sub_posts,
                    comment_limit=comments_per_post,
                    download_root=download_root,
                    source_keyword=keyword,
                    detailed_explanations=detailed_explanations,
                )
                all_results.extend(sub_results)
    else:
        for subreddit in subreddits:
            logger.info(f"Processing r/{subreddit}")
            posts = fetch_subreddit_posts(subreddit, posts_per_subreddit)
            sub_results = classify_reddit_content(
                predictor=predictor,
                subreddit=subreddit,
                posts=posts,
                comment_limit=comments_per_post,
                download_root=download_root,
                source_keyword=None,
                detailed_explanations=detailed_explanations,
            )
            all_results.extend(sub_results)

    save_results(all_results, output_path)
    logger.info(f"Completed. Saved {len(all_results)} predictions to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Reddit multimodal sarcasm inference pipeline")
    parser.add_argument('--subreddits', nargs='+', default=['AskReddit', 'funny', 'programming'])
    parser.add_argument('--keywords', nargs='*', default=None, help='Keyword list for Reddit search mode')
    parser.add_argument('--posts-per-subreddit', type=int, default=10)
    parser.add_argument('--comments-per-post', type=int, default=5)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model', type=str, default=None, help='Optional explicit model path')
    parser.add_argument('--output', type=str, default='results/reddit_multimodal_results.json')
    parser.add_argument('--detailed-explanations', action='store_true', help='Enable slower LIME/SHAP detailed explanations')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    keywords = args.keywords
    if keywords is not None and len(keywords) == 0:
        keywords = None

    # Interactive prompt mode if no keywords/subreddits explicitly provided
    if keywords is None:
        raw = input("Enter keywords separated by comma (leave empty to use subreddit mode): ").strip()
        if raw:
            keywords = [item.strip() for item in raw.split(',') if item.strip()]

    run_pipeline(
        keywords=keywords,
        subreddits=args.subreddits,
        posts_per_subreddit=args.posts_per_subreddit,
        comments_per_post=args.comments_per_post,
        config_path=args.config,
        model_path=args.model,
        output_path=args.output,
        detailed_explanations=args.detailed_explanations,
    )
