"""Backward-compatible entrypoint for Reddit multimodal sarcasm pipeline."""
from social_media_pipeline import run_pipeline


if __name__ == '__main__':
    run_pipeline(
        keywords=None,
        subreddits=['AskReddit', 'funny', 'programming'],
        posts_per_subreddit=10,
        comments_per_post=5,
        config_path='config/config.yaml',
        model_path=None,
        output_path='results/reddit_multimodal_results.json',
    )
