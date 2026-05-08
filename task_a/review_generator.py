"""
task_a/review_generator.py
---------------------------
The core Task A engine. Takes a UserPersona + item details and produces:
  - A predicted star rating (1–5, float)
  - A generated review text that sounds like THAT specific user

Architecture:
  1. Build a structured prompt from the persona (rating bias, style, examples)
  2. Ask the LLM to reason about the item given the user's known preferences
  3. Extract structured JSON output: {stars, review_text, reasoning}
  4. Apply Nigerian style post-processing if persona.use_naija_style is True

Key design decisions (for solution paper):
  - Chain-of-thought: LLM first reasons about the user's likely reaction,
    THEN generates the review. This improves rating accuracy.
  - Rating calibration: We anchor the predicted star to the user's avg_stars
    + item quality signals to fight the positive-skew bias (mean=3.84 in data)
  - Few-shot in-context examples: top-3 real reviews from the user's history
    are injected so the LLM mimics their actual writing voice.
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.persona import UserPersona
from shared.llm_client import LLMClient


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class GeneratedReview:
    stars: float            # predicted rating (1.0–5.0)
    stars_int: int          # rounded integer for RMSE comparison
    review_text: str        # generated review text
    reasoning: str          # LLM's internal reasoning (for ablation analysis)
    user_id: str
    item_name: str
    word_count: int


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at simulating authentic user reviews.
Your task is to generate a review that perfectly mimics a specific user's
writing voice, rating tendencies, and cultural style — as if THEY wrote it.

You always respond with valid JSON and nothing else.
No markdown, no preamble, no explanation outside the JSON structure."""


REVIEW_PROMPT_TEMPLATE = """{persona_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITEM TO REVIEW:
  Name:       {item_name}
  Category:   {item_category}
  Attributes: {item_attributes}
  Context:    {context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK:
Simulate how THIS specific user would rate and review the item above.

Think carefully about:
1. Does this item align with their top categories ({top_cats})?
2. Given their rating bias ({rating_bias}, avg {avg_stars:.1f}/5), how would they likely rate it?
3. Match their writing style: {writing_style} (~{target_words} words)
4. {naija_instruction}

IMPORTANT RATING CALIBRATION:
- This user's average rating is {avg_stars:.1f}/5
- If the item seems average for its category → rate close to {avg_stars:.1f}
- Only go significantly higher/lower if there's a clear reason from the item attributes
- Do NOT default to 4 or 5 stars just because they are common

Respond ONLY with this JSON structure:
{{
  "reasoning": "<2-3 sentences: why would this user react this way to this item?>",
  "stars": <float between 1.0 and 5.0, increments of 0.5>,
  "review_text": "<the review, written as if the user typed it>"
}}"""


NAIJA_STYLE_INSTRUCTIONS = {
    True: (
        "This user is Nigerian. Their review should naturally incorporate Nigerian English "
        "expressions and local cultural references where they fit organically. "
        "Examples: 'e good sha', 'value for money', 'the vibes was on point', "
        "'no be small thing', 'I nor go lie'. Don't force it — only where natural."
    ),
    False: "Write in standard English appropriate to the user's style.",
}


# ── Main generator ────────────────────────────────────────────────────────────

class ReviewGenerator:
    """
    Generates simulated reviews using a UserPersona and Groq LLM.
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(
            temperature=0.75,   # some creativity, not too random
            max_tokens=600,
            system_prompt=SYSTEM_PROMPT,
        )

    def generate(
        self,
        persona: UserPersona,
        item_name: str,
        item_category: str = "General",
        item_attributes: Optional[dict] = None,
        context: str = "",
    ) -> GeneratedReview:
        """
        Generate a review + rating for an item given a user persona.

        Args:
            persona:          UserPersona object
            item_name:        Name of the item/business
            item_category:    Category string (e.g. "Restaurants, Fast Food")
            item_attributes:  Dict of item attributes (price_range, delivery, etc.)
            context:          Optional context string (time of day, occasion, etc.)

        Returns:
            GeneratedReview with stars, review_text, and reasoning
        """
        attrs_str = self._format_attributes(item_attributes or {})
        target_words = self._target_word_count(persona)
        naija_instr  = NAIJA_STYLE_INSTRUCTIONS[persona.use_naija_style]

        prompt = REVIEW_PROMPT_TEMPLATE.format(
            persona_summary   = persona.to_prompt_summary(),
            item_name         = item_name,
            item_category     = item_category,
            item_attributes   = attrs_str,
            context           = context or "General visit",
            top_cats          = ", ".join(persona.top_categories[:3]) or "varied",
            rating_bias       = persona.rating_bias,
            avg_stars         = persona.avg_stars,
            writing_style     = persona.writing_style,
            target_words      = target_words,
            naija_instruction = naija_instr,
        )

        raw = self.llm.chat(prompt)
        parsed = self._parse_response(raw)

        stars_float = float(parsed.get("stars", persona.avg_stars))
        stars_float = max(1.0, min(5.0, stars_float))
        stars_float = round(stars_float * 2) / 2  # snap to 0.5 increments

        review_text = str(parsed.get("review_text", "")).strip()
        reasoning   = str(parsed.get("reasoning", "")).strip()

        return GeneratedReview(
            stars       = stars_float,
            stars_int   = round(stars_float),
            review_text = review_text,
            reasoning   = reasoning,
            user_id     = persona.user_id,
            item_name   = item_name,
            word_count  = len(review_text.split()),
        )

    def generate_batch(
        self,
        persona: UserPersona,
        items: list[dict],
    ) -> list[GeneratedReview]:
        """
        Generate reviews for multiple items for the same persona.
        items: list of dicts with keys: name, category, attributes, context
        """
        results = []
        for item in items:
            result = self.generate(
                persona          = persona,
                item_name        = item["name"],
                item_category    = item.get("category", "General"),
                item_attributes  = item.get("attributes", {}),
                context          = item.get("context", ""),
            )
            results.append(result)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _target_word_count(self, persona: UserPersona) -> int:
        """Return target word count based on persona's writing style."""
        style_targets = {
            "concise":    50,
            "neutral":    100,
            "detailed":   150,
            "verbose":    200,
        }
        return style_targets.get(persona.writing_style, int(persona.avg_word_count))

    def _format_attributes(self, attrs: dict) -> str:
        if not attrs:
            return "Not specified"
        parts = []
        labels = {
            "price_range": lambda v: "$" * int(v) if isinstance(v, (int, float)) else str(v),
            "delivery":    lambda v: "Delivery available" if v else "Dine-in only",
            "wifi":        lambda v: "Free WiFi" if v else "No WiFi",
            "parking":     lambda v: f"Parking: {v}",
            "noise_level": lambda v: f"Noise: {v}",
            "attire":      lambda v: f"Dress code: {v}",
        }
        for k, v in attrs.items():
            if k in labels:
                parts.append(labels[k](v))
            else:
                parts.append(f"{k}: {v}")
        return ", ".join(parts)

    def _parse_response(self, raw: str) -> dict:
        """
        Robustly parse the LLM's JSON response.
        Falls back gracefully if the model adds markdown fences.
        """
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Handle case where model wraps in outer text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: extract fields with regex
            stars_match = re.search(r'"stars"\s*:\s*([0-9.]+)', cleaned)
            text_match  = re.search(r'"review_text"\s*:\s*"(.+?)"(?=\s*[,}])', cleaned, re.DOTALL)
            reason_match = re.search(r'"reasoning"\s*:\s*"(.+?)"(?=\s*[,}])', cleaned, re.DOTALL)
            return {
                "stars":       float(stars_match.group(1)) if stars_match else 3.5,
                "review_text": text_match.group(1) if text_match else raw[:500],
                "reasoning":   reason_match.group(1) if reason_match else "",
            }