"""
task_a/app.py
-------------
FastAPI application for Task A — User Modeling & Review Generation.

Endpoints:
  POST /generate-review      → generate one review from persona + item
  POST /generate-review/batch → generate reviews for multiple items
  POST /generate-review/from-history → build persona from provided history then generate
  GET  /health               → health check
  GET  /sample-persona       → returns a sample persona from real data (for demo)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import traceback

from shared.persona import UserPersona
from shared.llm_client import LLMClient
from task_a.persona_builder import (
    build_from_user_id,
    build_from_raw,
    get_random_user_id,
    get_test_reviews_for_user,
)
from task_a.review_generator import ReviewGenerator


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Root & Rise — Task A: User Modeling",
    description=(
        "Simulates user reviews and star ratings for unseen items "
        "by modeling individual user behaviour, tone, and preferences.\n\n"
        "Built for DSN x BCT LLM Agent Challenge 2026 by Root & Rise."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton LLM client and generator (shared across requests)
_llm = LLMClient(temperature=0.75, max_tokens=600)
_generator = ReviewGenerator(llm=_llm)


# ── Request / Response schemas ─────────────────────────────────────────────────

class ItemDetails(BaseModel):
    name: str = Field(..., example="Chicken Republic, Lekki")
    category: str = Field("Restaurants", example="Fast Food, Restaurants")
    attributes: dict = Field(default_factory=dict, example={"price_range": 2, "delivery": True})
    context: str = Field("", example="Lunch on a weekday")


class PersonaInput(BaseModel):
    """Structured persona for direct injection (no DB lookup needed)."""
    user_id: str
    avg_stars: float = Field(3.8, ge=1.0, le=5.0)
    rating_std: float = 1.0
    review_count: int = 10
    writing_style: str = "neutral"   # concise | neutral | detailed | verbose
    avg_word_count: float = 100.0
    top_categories: list[str] = Field(default_factory=list)
    rating_bias: str = "balanced"    # harsh | balanced | generous
    use_naija_style: bool = True
    location_context: str = "Nigeria"
    sample_reviews: list[dict] = Field(
        default_factory=list,
        description="Optional: list of past reviews {stars, text, business_name}"
    )


class ReviewFromUserIdRequest(BaseModel):
    user_id: str
    item: ItemDetails
    use_naija_style: bool = True


class ReviewFromPersonaRequest(BaseModel):
    user_persona: PersonaInput
    item: ItemDetails


class ReviewFromHistoryRequest(BaseModel):
    user_id: str
    review_history: list[dict] = Field(
        ...,
        description="List of past reviews: [{stars, text, business_name, categories}]"
    )
    item: ItemDetails
    use_naija_style: bool = True
    location_context: str = "Nigeria"


class BatchReviewRequest(BaseModel):
    user_persona: PersonaInput
    items: list[ItemDetails]


class ReviewResponse(BaseModel):
    user_id: str
    item_name: str
    predicted_stars: float
    predicted_stars_int: int
    review_text: str
    reasoning: str
    word_count: int
    persona_summary: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "task_a", "model": _llm.model}


@app.get("/sample-persona")
def sample_persona():
    """Returns a random real user's persona for demo/testing purposes."""
    try:
        uid = get_random_user_id(min_reviews=10)
        persona = build_from_user_id(uid, use_naija_style=True)
        return {
            "user_id":         persona.user_id,
            "avg_stars":       persona.avg_stars,
            "rating_bias":     persona.rating_bias,
            "writing_style":   persona.writing_style,
            "avg_word_count":  persona.avg_word_count,
            "top_categories":  persona.top_categories,
            "review_count":    persona.review_count,
            "sample_reviews":  persona.sample_reviews[:2],
            "persona_summary": persona.to_prompt_summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-review/from-user-id", response_model=ReviewResponse)
def generate_from_user_id(req: ReviewFromUserIdRequest):
    """
    Look up a real Yelp user by ID, build their persona from training history,
    and generate a review for the given item.
    """
    try:
        persona = build_from_user_id(req.user_id, use_naija_style=req.use_naija_style)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return _run_generation(persona, req.item)


@app.post("/generate-review", response_model=ReviewResponse)
def generate_from_persona(req: ReviewFromPersonaRequest):
    """
    Generate a review using a fully specified persona object.
    No database lookup — works for any user, real or hypothetical.
    This is the primary endpoint for the hackathon demo.
    """
    persona = _persona_from_input(req.user_persona)
    return _run_generation(persona, req.item)


@app.post("/generate-review/from-history", response_model=ReviewResponse)
def generate_from_history(req: ReviewFromHistoryRequest):
    """
    Build a persona on-the-fly from provided review history, then generate.
    Useful for users not in the Yelp dataset.
    """
    if not req.review_history:
        raise HTTPException(status_code=400, detail="review_history cannot be empty")

    persona = build_from_raw(
        user_id          = req.user_id,
        reviews          = req.review_history,
        use_naija_style  = req.use_naija_style,
        location_context = req.location_context,
    )
    return _run_generation(persona, req.item)


@app.post("/generate-review/batch")
def generate_batch(req: BatchReviewRequest):
    """Generate reviews for multiple items for the same persona."""
    persona = _persona_from_input(req.user_persona)
    items   = [i.model_dump() for i in req.items]
    results = _generator.generate_batch(persona, items)

    return {
        "user_id": persona.user_id,
        "reviews": [
            {
                "item_name":       r.item_name,
                "predicted_stars": r.stars,
                "review_text":     r.review_text,
                "word_count":      r.word_count,
            }
            for r in results
        ],
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _persona_from_input(p: PersonaInput) -> UserPersona:
    persona = UserPersona(
        user_id            = p.user_id,
        avg_stars          = p.avg_stars,
        rating_std         = p.rating_std,
        review_count       = p.review_count,
        writing_style      = p.writing_style,
        avg_word_count     = p.avg_word_count,
        top_categories     = p.top_categories,
        rating_bias        = p.rating_bias,
        use_naija_style    = p.use_naija_style,
        location_context   = p.location_context,
        sample_reviews     = p.sample_reviews,
    )
    return persona


def _run_generation(persona: UserPersona, item: ItemDetails) -> ReviewResponse:
    try:
        result = _generator.generate(
            persona          = persona,
            item_name        = item.name,
            item_category    = item.category,
            item_attributes  = item.attributes,
            context          = item.context,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return ReviewResponse(
        user_id             = result.user_id,
        item_name           = result.item_name,
        predicted_stars     = result.stars,
        predicted_stars_int = result.stars_int,
        review_text         = result.review_text,
        reasoning           = result.reasoning,
        word_count          = result.word_count,
        persona_summary     = persona.to_prompt_summary(),
    )