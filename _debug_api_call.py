from api_server import predict_by_keywords, KeywordRequest

payload = KeywordRequest(
    keywords=["Donald trump"],
    posts_per_subreddit=1,
    comments_per_post=0,
    detailed_explanations=False,
)

result = predict_by_keywords(payload)
print("TOTAL", result.total_results)
