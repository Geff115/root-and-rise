"""
shared/persona.py
-----------------
UserPersona data model — the central object shared across Task A and Task B.
A persona is built from a user's review history and captures:
  - Statistical signals (avg rating, variance, review frequency)
  - Stylistic signals (avg word count, vocabulary richness, sentiment)
  - Preference signals (top categories, top businesses)
  - Contextual signals (time-of-day patterns, seasonal behaviour)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserPersona:
    """
    Represents a user's behavioural and stylistic profile
    derived from their review history.
    """

    # --- Identity ---
    user_id: str
    name: Optional[str] = None

    # --- Rating Behaviour ---
    avg_stars: float = 3.5          # Mean rating given
    rating_std: float = 1.0         # Std deviation (high = volatile rater)
    rating_distribution: dict = field(default_factory=lambda: {
        1: 0, 2: 0, 3: 0, 4: 0, 5: 0
    })
    # Bias: "harsh" (avg < 3.0), "generous" (avg > 4.0), "balanced"
    rating_bias: str = "balanced"

    # --- Writing Style ---
    review_count: int = 0
    avg_word_count: float = 80.0    # Average words per review
    vocabulary_richness: float = 0.5  # Type-token ratio (0–1)
    avg_sentiment: float = 0.0      # Compound VADER sentiment (-1 to +1)
    writing_style: str = "neutral"  # e.g. "concise", "verbose", "storytelling"

    # --- Preferences ---
    top_categories: list = field(default_factory=list)   # e.g. ["restaurants", "bars"]
    top_businesses: list = field(default_factory=list)   # Most reviewed businesses
    preferred_price_range: int = 2  # 1 (cheap) to 4 (expensive)

    # --- Contextual ---
    most_active_hour: Optional[int] = None   # Hour of day (0–23)
    most_active_day: Optional[str] = None    # Day of week
    years_active: float = 1.0

    # --- Nigerian Context ---
    # Optional flag to activate Nigerian English/Pidgin style injection
    use_naija_style: bool = False
    location_context: Optional[str] = None  # e.g. "Lagos", "Abuja", "PH"

    # --- Raw history (for in-context examples) ---
    sample_reviews: list = field(default_factory=list)
    # Each item: {"business": str, "stars": int, "text": str, "date": str}

    def to_prompt_summary(self) -> str:
        """
        Serialises the persona into a human-readable summary
        suitable for injection into an LLM prompt.
        """
        bias_map = {
            "harsh":     "tends to rate critically — their 4-star is most people's 5-star",
            "generous":  "tends to rate generously — rarely goes below 3 stars",
            "balanced":  "gives ratings that closely reflect their actual experience",
        }
        style_desc = {
            "concise":      "writes short, punchy reviews (under 60 words)",
            "verbose":      "writes long, detailed reviews (150+ words)",
            "storytelling": "narrates their experience like a story",
            "neutral":      "writes straightforward, factual reviews",
        }
        naija_note = ""
        if self.use_naija_style:
            loc = self.location_context or "Nigeria"
            naija_note = (
                f"\n- Cultural context: User is based in {loc}. "
                "Naturally incorporates Nigerian English expressions and local references."
            )

        sample_text = ""
        if self.sample_reviews:
            examples = self.sample_reviews[:3]
            sample_text = "\n\nSAMPLE REVIEWS FROM THIS USER:\n"
            for i, r in enumerate(examples, 1):
                sample_text += (
                    f"\n[Example {i}] {r.get('business','?')} — "
                    f"{r.get('stars','?')}★\n\"{r.get('text','')}\"\n"
                )

        return f"""USER PROFILE:
- User ID: {self.user_id}
- Total reviews written: {self.review_count}
- Average star rating given: {self.avg_stars:.1f}/5.0 (std: {self.rating_std:.2f})
- Rating tendency: {bias_map.get(self.rating_bias, self.rating_bias)}
- Writing style: {style_desc.get(self.writing_style, self.writing_style)}
- Average review length: ~{int(self.avg_word_count)} words
- Top categories they review: {', '.join(self.top_categories[:5]) or 'varied'}
- Preferred price range: {'$' * self.preferred_price_range} (1=budget, 4=luxury)
- Sentiment profile: {'positive' if self.avg_sentiment > 0.1 else 'negative' if self.avg_sentiment < -0.1 else 'neutral'} leaning{naija_note}{sample_text}"""

    @classmethod
    def from_review_history(cls, user_id: str, reviews: list[dict]) -> "UserPersona":
        """
        Build a UserPersona from a list of raw review dicts.
        Each review should have: {stars, text, business_name, date, categories}
        """
        import numpy as np
        from collections import Counter

        if not reviews:
            return cls(user_id=user_id)

        stars = [r["stars"] for r in reviews]
        avg = float(np.mean(stars))
        std = float(np.std(stars))

        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for s in stars:
            dist[int(s)] = dist.get(int(s), 0) + 1

        if avg < 3.0:
            bias = "harsh"
        elif avg > 4.0:
            bias = "generous"
        else:
            bias = "balanced"

        word_counts = [len(r["text"].split()) for r in reviews if r.get("text")]
        avg_wc = float(np.mean(word_counts)) if word_counts else 80.0

        if avg_wc < 60:
            style = "concise"
        elif avg_wc > 150:
            style = "verbose"
        else:
            style = "neutral"

        all_categories = []
        for r in reviews:
            all_categories.extend(r.get("categories", []))
        top_cats = [cat for cat, _ in Counter(all_categories).most_common(5)]

        top_biz = [
            biz for biz, _ in
            Counter(r.get("business_name", "") for r in reviews).most_common(5)
        ]

        sample = [
            {
                "business": r.get("business_name", ""),
                "stars": r.get("stars", 0),
                "text": r.get("text", "")[:300],  # truncate for prompt safety
                "date": r.get("date", ""),
            }
            for r in sorted(reviews, key=lambda x: x.get("date", ""), reverse=True)[:5]
        ]

        return cls(
            user_id=user_id,
            review_count=len(reviews),
            avg_stars=round(avg, 2),
            rating_std=round(std, 2),
            rating_distribution=dist,
            rating_bias=bias,
            avg_word_count=round(avg_wc, 1),
            writing_style=style,
            top_categories=top_cats,
            top_businesses=top_biz,
            sample_reviews=sample,
        )