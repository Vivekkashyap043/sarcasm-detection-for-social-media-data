"""FastAPI service for frontend-driven Reddit sarcasm inference."""
import logging
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import MultimodalSarcasmPredictor
from social_media_pipeline import collect_pipeline_results, save_results


app = FastAPI(title="Sarcasm Detection API", version="1.0.0")
logger = logging.getLogger(__name__)


class KeywordRequest(BaseModel):
    keywords: List[str] = Field(default_factory=list, description="Keyword list to search on Reddit")
    subreddits: List[str] = Field(default_factory=lambda: ["AskReddit", "funny", "programming"])
    posts_per_subreddit: int = Field(default=10, ge=1, le=100)
    comments_per_post: int = Field(default=5, ge=0, le=50)
    detailed_explanations: bool = False


class PredictionResponse(BaseModel):
    total_results: int
    results: list


_predictor: Optional[MultimodalSarcasmPredictor] = None
DEFAULT_OUTPUT_PATH = "results/reddit_multimodal_results.json"
API_BUILD = "2026-03-06-api-fix-2"


def get_predictor() -> MultimodalSarcasmPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MultimodalSarcasmPredictor(config_path="config/config.yaml", model_path=None)
    return _predictor


@app.get("/health")
def health_check():
    return {"status": "ok", "build": API_BUILD}


@app.post("/predict/keywords", response_model=PredictionResponse)
def predict_by_keywords(payload: KeywordRequest):
    predictor = get_predictor()
    output_path = DEFAULT_OUTPUT_PATH

    keywords = [k.strip() for k in payload.keywords if k and k.strip()]
    if not keywords:
        keywords = None

    all_results_raw = collect_pipeline_results(
        predictor=predictor,
        keywords=keywords,
        subreddits=payload.subreddits,
        posts_per_subreddit=payload.posts_per_subreddit,
        comments_per_post=payload.comments_per_post,
        output_path=output_path,
        detailed_explanations=payload.detailed_explanations,
    )

    if all_results_raw is None:
        all_results = []
    elif isinstance(all_results_raw, list):
        all_results = all_results_raw
    else:
        all_results = list(all_results_raw)

    if not all_results:
        logger.warning("Keyword prediction completed with no results returned.")

    save_results(all_results, output_path)

    return PredictionResponse(
        total_results=len(all_results),
        results=all_results,
    )
