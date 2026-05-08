"""
task_a/persona_builder.py
--------------------------
Loads a user's review history from the processed Yelp parquet files
and constructs a rich UserPersona object ready for the review generator.

Two entry points:
  - build_from_user_id()  → looks up a real user from processed data
  - build_from_raw()      → builds from a list of review dicts (API input)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from collections import Counter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.persona import UserPersona

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# ── Data loaders (cached so we don't re-read parquet on every request) ──────

@lru_cache(maxsize=1)
def _load_reviews() -> pd.DataFrame:
    path = DATA_DIR / "reviews_split.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}.\n"
            "Run notebooks/01_data_exploration.ipynb first to generate it."
        )
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_businesses() -> pd.DataFrame:
    path = DATA_DIR / "businesses.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ── Core builders ─────────────────────────────────────────────────────────────

def build_from_user_id(
    user_id: str,
    split: str = "train",
    use_naija_style: bool = True,
    location_context: str = "Nigeria",
) -> UserPersona:
    """
    Build a UserPersona from processed Yelp data for a known user_id.

    Args:
        user_id:           Yelp user_id string
        split:             Which split to use for persona construction ('train')
        use_naija_style:   Activate Nigerian English prompt injection
        location_context:  City hint for cultural grounding

    Returns:
        UserPersona with all fields populated
    """
    df = _load_reviews()
    df_biz = _load_businesses()

    user_reviews = df[(df["user_id"] == user_id) & (df["split"] == split)].copy()

    if user_reviews.empty:
        raise ValueError(f"No '{split}' reviews found for user_id='{user_id}'")

    # Merge in business metadata if available
    if not df_biz.empty and "business_id" in user_reviews.columns:
        user_reviews = user_reviews.merge(
            df_biz[["business_id", "name", "categories"]].rename(
                columns={"name": "biz_name"}
            ),
            on="business_id",
            how="left",
        )
    else:
        user_reviews["biz_name"] = "Unknown"
        user_reviews["categories"] = ""

    # Build raw review dicts for the shared persona builder
    raw_reviews = []
    for _, row in user_reviews.iterrows():
        raw_reviews.append({
            "stars":         int(row["stars"]),
            "text":          str(row.get("text", "")),
            "business_name": str(row.get("biz_name", "")),
            "date":          str(row.get("date", "")),
            "categories":    [
                c.strip()
                for c in str(row.get("categories", "")).split(",")
                if c.strip()
            ],
        })

    persona = UserPersona.from_review_history(user_id, raw_reviews)
    persona.use_naija_style = use_naija_style
    persona.location_context = location_context
    return persona


def build_from_raw(
    user_id: str,
    reviews: list[dict],
    use_naija_style: bool = True,
    location_context: str = "Nigeria",
) -> UserPersona:
    """
    Build a UserPersona from a list of review dicts provided directly
    (e.g. from the API request body — no database lookup needed).

    Each review dict should have:
        stars (int), text (str), business_name (str),
        date (str, optional), categories (list[str], optional)
    """
    persona = UserPersona.from_review_history(user_id, reviews)
    persona.use_naija_style = use_naija_style
    persona.location_context = location_context
    return persona


def get_random_user_id(min_reviews: int = 10) -> str:
    """Return a random user_id with enough reviews for a meaningful persona."""
    df = _load_reviews()
    counts = df[df["split"] == "train"]["user_id"].value_counts()
    eligible = counts[counts >= min_reviews].index.tolist()
    if not eligible:
        raise ValueError("No users found with enough reviews. Check processed data.")
    return np.random.choice(eligible)


def get_test_reviews_for_user(user_id: str) -> list[dict]:
    """
    Return the test-split reviews for a user.
    Used during evaluation: we compare generated reviews against these.
    """
    df = _load_reviews()
    df_biz = _load_businesses()

    test_rows = df[(df["user_id"] == user_id) & (df["split"] == "test")].copy()

    if not df_biz.empty:
        test_rows = test_rows.merge(
            df_biz[["business_id", "name", "categories"]].rename(
                columns={"name": "biz_name"}
            ),
            on="business_id",
            how="left",
        )
    else:
        test_rows["biz_name"] = "Unknown"
        test_rows["categories"] = ""

    return [
        {
            "business_id":   row["business_id"],
            "business_name": str(row.get("biz_name", "")),
            "categories":    str(row.get("categories", "")),
            "actual_stars":  int(row["stars"]),
            "actual_text":   str(row.get("text", "")),
        }
        for _, row in test_rows.iterrows()
    ]