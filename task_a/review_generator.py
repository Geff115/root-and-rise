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
  4. Nigerian style is MANDATORY when use_naija_style=True — not optional flavour

Key design decisions (for solution paper):
  - Chain-of-thought: LLM reasons about the user's likely reaction FIRST,
    then generates the review. This improves rating accuracy.
  - Rating calibration: We anchor predicted stars to user's personal avg_stars
    to fight the positive-skew bias (mean=3.84 observed in Yelp data).
  - Nigerian authenticity layer: When use_naija_style=True, the prompt enforces
    Nigerian English and Pidgin patterns as a hard requirement, not a suggestion.
    This is a primary scoring differentiator in this competition.
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
    stars: float
    stars_int: int
    review_text: str
    reasoning: str
    user_id: str
    item_name: str
    word_count: int


# ── Nigerian language toolkit ─────────────────────────────────────────────────
# This is injected directly into the prompt when use_naija_style=True.
# Richness here = higher behavioural fidelity score from human evaluators.

NAIJA_LANGUAGE_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║              NIGERIAN VOICE — MANDATORY REQUIREMENTS             ║
╚══════════════════════════════════════════════════════════════════╝

This review MUST sound authentically Nigerian. This is not optional flavour —
it is a core requirement. The review should feel like it was written by a real
Nigerian person living in Nigeria, expressing themselves naturally.

MANDATORY: Use AT LEAST 3 of the following authentic patterns:

1. NIGERIAN PIDGIN ENGLISH phrases (use naturally, not forced):
   - "I nor go lie" / "no be lie"          → "honestly" / "I'm not gonna lie"
   - "e good sha" / "e sweet well well"    → "it's good" / "it's really great"
   - "the thing dey" / "e dey there"       → "it exists" / "it's there"
   - "na so e be" / "na so dem do am"      → "that's how it is"
   - "abeg" / "abeg o"                     → "please" / expression of emphasis
   - "omo" / "omo mehn"                    → expression of surprise/emphasis
   - "e don do" / "e don finish"           → "it's done" / "they ran out"
   - "wetin" / "wetin concern me"          → "what" / "what does it matter"
   - "no be small thing"                   → "it's not a small deal"
   - "I go come back" / "I go dey"         → "I'll return" / "I'll be here"

2. NIGERIAN ENGLISH expressions (standard Nigerian English, not Pidgin):
   - "value for money"                     → very common Nigerian phrase
   - "the vibes was on point"              → atmosphere was great
   - "they really tried"                   → they made a good effort
   - "service delivery"                    → quality of service
   - "the environment is conducive"        → the place is comfortable
   - "I must commend them"                 → I want to compliment them
   - "they kept me waiting"                → I waited a long time
   - "it is what it is sha"               → resigned acceptance
   - "nothing to write home about"        → unremarkable
   - "sharp sharp" / "quick quick"        → very fast
   - "I was not disappointed"             → met expectations

3. NIGERIAN CULTURAL REFERENCES (where relevant to context):
   - Compare food to familiar Nigerian benchmarks
     e.g. "almost as good as mama's jollof", "reminded me of Bukka Hut"
   - Reference local price consciousness: "for Lagos price, e make sense"
   - Reference Nigerian lifestyle: traffic, NEPA/power, generator noise
   - Local food references where fitting: puff puff, suya, jollof, pepper soup
   - Local brands as comparison points: Chicken Republic, Mr Biggs, Tantalizers

4. TONE characteristics of Nigerian reviewers:
   - Direct and opinionated — Nigerians don't hedge excessively
   - Communal framing — "my people", "they", referring to staff warmly or critically
   - Value-conscious — always weighing quality vs price
   - Expressive disappointment when let down ("this one pain me", "I was not happy at all")
   - Warm praise when impressed ("they deserve all the stars", "God bless the chef")

IMPORTANT: Integrate these naturally into the review. Do NOT make a list of
Pidgin phrases. Write like a real person thinking and typing, not performing.
The Nigerian voice should feel organic, the way code-switching happens naturally.
"""


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at simulating authentic user reviews.
Your task is to generate a review that perfectly mimics a specific user's
writing voice, rating tendencies, and cultural identity — as if THEY wrote it.

When the user is Nigerian, you write in an authentically Nigerian voice,
naturally blending Nigerian English, Pidgin phrases, and cultural references.
This is not a stylistic flourish — it is who this person IS.

You always respond with valid JSON and nothing else.
No markdown, no preamble, no explanation outside the JSON structure."""


REVIEW_PROMPT_TEMPLATE = """{persona_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITEM TO REVIEW:
  Name:       {item_name}
  Category:   {item_category}
  Attributes: {item_attributes}
  Context:    {context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{naija_block}

TASK — Simulate how THIS specific user would rate and review the item above.

STEP 1 — REASON first (this goes in "reasoning"):
  - Does this item align with their top categories ({top_cats})?
  - Given their {rating_bias} rating tendency (avg {avg_stars:.1f}/5), what would they rate it?
  - What aspects would they focus on given their writing style?

STEP 2 — GENERATE the review:
  - Match their writing style: {writing_style} (~{target_words} words target)
  - Mirror the voice and sentence patterns from their sample reviews above
  - Apply all Nigerian voice requirements if specified above

RATING CALIBRATION (critical for accuracy):
  - This user's personal average is {avg_stars:.1f}/5 — use this as your anchor
  - An "average" experience for this item → rate near {avg_stars:.1f}
  - Only go significantly higher/lower if the item attributes clearly justify it
  - NEVER default to 4 or 5 stars just because high ratings are common

Respond ONLY with this JSON:
{{
  "reasoning": "<2-3 sentences: why this user reacts this way to this item>",
  "stars": <float 1.0–5.0 in 0.5 increments>,
  "review_text": "<the review, written exactly as the user would type it>"
}}"""


# ── Main generator ────────────────────────────────────────────────────────────

class ReviewGenerator:

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(
            temperature=0.78,
            max_tokens=700,
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

        attrs_str    = self._format_attributes(item_attributes or {})
        target_words = self._target_word_count(persona)

        # Nigerian block is MANDATORY content, not a soft suggestion
        naija_block = NAIJA_LANGUAGE_GUIDE if persona.use_naija_style else (
            "Write in standard English appropriate to the user's established style."
        )

        prompt = REVIEW_PROMPT_TEMPLATE.format(
            persona_summary  = persona.to_prompt_summary(),
            item_name        = item_name,
            item_category    = item_category,
            item_attributes  = attrs_str,
            context          = context or "General visit",
            naija_block      = naija_block,
            top_cats         = ", ".join(persona.top_categories[:3]) or "varied",
            rating_bias      = persona.rating_bias,
            avg_stars        = persona.avg_stars,
            writing_style    = persona.writing_style,
            target_words     = target_words,
        )

        raw    = self.llm.chat(prompt)
        parsed = self._parse_response(raw)

        stars_float = float(parsed.get("stars", persona.avg_stars))
        stars_float = max(1.0, min(5.0, stars_float))
        stars_float = round(stars_float * 2) / 2  # snap to 0.5 increments

        review_text = str(parsed.get("review_text", "")).strip()
        reasoning   = str(parsed.get("reasoning", "")).strip()

        # Post-generation Nigerian voice validation
        if persona.use_naija_style:
            review_text = self._enforce_naija_voice(review_text, persona, item_name)

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
        results = []
        for item in items:
            result = self.generate(
                persona         = persona,
                item_name       = item["name"],
                item_category   = item.get("category", "General"),
                item_attributes = item.get("attributes", {}),
                context         = item.get("context", ""),
            )
            results.append(result)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enforce_naija_voice(
        self, review_text: str, persona: UserPersona, item_name: str
    ) -> str:
        """
        If the generated review doesn't contain enough Nigerian markers,
        make a second targeted LLM call to rewrite it with stronger Nigerian voice.
        This is our quality gate for the Nigerian scoring criterion.
        """
        naija_markers = [
            "sha", "omo", "abeg", "nor", "dey", "wetin", "mehn",
            "e good", "no be", "na so", "sharp sharp", "value for money",
            "the vibes", "they tried", "conducive", "commend",
        ]
        marker_count = sum(1 for m in naija_markers if m.lower() in review_text.lower())

        if marker_count >= 2:
            return review_text  # already sufficiently Nigerian

        # Second-pass rewrite with explicit Nigerian focus
        rewrite_prompt = f"""The following review was written by a Nigerian person but doesn't 
sound authentically Nigerian enough. Rewrite it to naturally incorporate Nigerian English 
and Pidgin expressions while keeping the same meaning, sentiment, rating, and approximate length.

ORIGINAL REVIEW:
\"{review_text}\"

ITEM: {item_name}
USER'S WRITING STYLE: {persona.writing_style} (~{int(persona.avg_word_count)} words)

Rules:
- Keep the same sentiment and core message
- Add at least 3 natural Nigerian expressions (Pidgin or Nigerian English)  
- Do NOT make it sound forced or like a caricature
- Return ONLY the rewritten review text, no JSON, no preamble"""

        rewritten = self.llm.chat(rewrite_prompt, temperature=0.7)
        # Strip any accidental JSON wrapping
        rewritten = re.sub(r'^["\'`]|["\'`]$', '', rewritten.strip())
        return rewritten if len(rewritten) > 30 else review_text

    def _target_word_count(self, persona: UserPersona) -> int:
        style_targets = {
            "concise":  50,
            "neutral":  100,
            "detailed": 150,
            "verbose":  200,
        }
        return style_targets.get(persona.writing_style, int(persona.avg_word_count))

    def _format_attributes(self, attrs: dict) -> str:
        if not attrs:
            return "Not specified"
        parts = []
        labels = {
            "price_range": lambda v: "$" * int(v),
            "delivery":    lambda v: "Delivery available" if v else "Dine-in only",
            "wifi":        lambda v: "Free WiFi" if v else "No WiFi",
            "parking":     lambda v: f"Parking: {v}",
            "noise_level": lambda v: f"Noise: {v}",
            "attire":      lambda v: f"Dress code: {v}",
            "reservation": lambda v: "Reservations accepted" if v else "Walk-in only",
        }
        for k, v in attrs.items():
            parts.append(labels[k](v) if k in labels else f"{k}: {v}")
        return ", ".join(parts)

    def _parse_response(self, raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            stars_m  = re.search(r'"stars"\s*:\s*([0-9.]+)', cleaned)
            text_m   = re.search(r'"review_text"\s*:\s*"(.+?)"(?=\s*[,}])', cleaned, re.DOTALL)
            reason_m = re.search(r'"reasoning"\s*:\s*"(.+?)"(?=\s*[,}])', cleaned, re.DOTALL)
            return {
                "stars":       float(stars_m.group(1)) if stars_m else 3.5,
                "review_text": text_m.group(1) if text_m else raw[:500],
                "reasoning":   reason_m.group(1) if reason_m else "",
            }