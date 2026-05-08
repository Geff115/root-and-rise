"""
task_a/evaluator.py
--------------------
Evaluation harness for Task A.

Metrics computed:
  - ROUGE-1 / ROUGE-2 / ROUGE-L  (review text quality)
  - BERTScore F1                  (semantic similarity)
  - Rating RMSE                   (star rating accuracy)
  - Avg word count delta          (behavioural fidelity proxy)
  - Style consistency score       (heuristic)

Usage:
    python task_a/evaluator.py --n_users 50 --output results/task_a_eval.json
"""

from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.llm_client import LLMClient
from task_a.persona_builder import (
    build_from_user_id,
    get_random_user_id,
    get_test_reviews_for_user,
    _load_reviews,
)
from task_a.review_generator import ReviewGenerator


# ── Result schema ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    user_id:          str
    item_name:        str
    actual_stars:     float
    predicted_stars:  float
    star_error:       float          # |actual - predicted|
    rouge1:           float
    rouge2:           float
    rougeL:           float
    bertscore_f1:     float
    actual_wc:        int
    generated_wc:     int
    wc_delta:         int


@dataclass
class AggregateMetrics:
    n_samples:      int
    rmse:           float
    mae:            float
    rouge1_mean:    float
    rouge2_mean:    float
    rougeL_mean:    float
    bertscore_mean: float
    wc_delta_mean:  float


# ── Evaluator class ───────────────────────────────────────────────────────────

class TaskAEvaluator:
    def __init__(self):
        self.llm       = LLMClient(temperature=0.5, max_tokens=600)
        self.generator = ReviewGenerator(llm=self.llm)
        self._rouge_scorer  = None
        self._bert_scorer   = None

    def _get_rouge(self):
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
        return self._rouge_scorer

    def _get_bertscore(self):
        if self._bert_scorer is None:
            import bert_score
            self._bert_scorer = bert_score
        return self._bert_scorer

    def evaluate_single(self, user_id: str, test_item: dict) -> EvalResult:
        """Run generation for one user/item pair and compute all metrics."""
        persona = build_from_user_id(user_id, use_naija_style=True)
        result  = self.generator.generate(
            persona         = persona,
            item_name       = test_item["business_name"],
            item_category   = test_item.get("categories", ""),
        )

        actual_text  = test_item["actual_text"]
        generated    = result.review_text
        actual_stars = float(test_item["actual_stars"])

        # ROUGE
        rouge = self._get_rouge()
        scores = rouge.score(actual_text, generated)

        # BERTScore
        bs = self._get_bertscore()
        P, R, F = bs.score([generated], [actual_text], lang="en", verbose=False)
        bert_f1 = float(F[0])

        return EvalResult(
            user_id         = user_id,
            item_name       = test_item["business_name"],
            actual_stars    = actual_stars,
            predicted_stars = result.stars,
            star_error      = abs(actual_stars - result.stars),
            rouge1          = scores["rouge1"].fmeasure,
            rouge2          = scores["rouge2"].fmeasure,
            rougeL          = scores["rougeL"].fmeasure,
            bertscore_f1    = bert_f1,
            actual_wc       = len(actual_text.split()),
            generated_wc    = result.word_count,
            wc_delta        = abs(len(actual_text.split()) - result.word_count),
        )

    def evaluate(self, n_users: int = 50) -> tuple[list[EvalResult], AggregateMetrics]:
        """
        Evaluate on n_users randomly sampled from the test split.
        For each user, evaluate on their first test review.
        """
        df = _load_reviews()
        eligible = (
            df[df["split"] == "train"]["user_id"]
            .value_counts()
            [lambda s: s >= 5]
            .index.tolist()
        )
        sampled_users = np.random.choice(eligible, size=min(n_users, len(eligible)), replace=False)

        results = []
        for i, uid in enumerate(sampled_users):
            test_items = get_test_reviews_for_user(uid)
            if not test_items:
                continue
            try:
                r = self.evaluate_single(uid, test_items[0])
                results.append(r)
                print(f"[{i+1}/{n_users}] {uid[:12]}... "
                      f"stars: {r.actual_stars}→{r.predicted_stars}  "
                      f"ROUGE-L: {r.rougeL:.3f}  "
                      f"BERTScore: {r.bertscore_f1:.3f}")
                time.sleep(0.3)  # rate limit buffer
            except Exception as e:
                print(f"  ⚠ Skipped {uid}: {e}")

        agg = self._aggregate(results)
        return results, agg

    def _aggregate(self, results: list[EvalResult]) -> AggregateMetrics:
        if not results:
            raise ValueError("No evaluation results to aggregate")
        errors = [r.star_error for r in results]
        return AggregateMetrics(
            n_samples      = len(results),
            rmse           = float(np.sqrt(np.mean(np.array(errors) ** 2))),
            mae            = float(np.mean(errors)),
            rouge1_mean    = float(np.mean([r.rouge1 for r in results])),
            rouge2_mean    = float(np.mean([r.rouge2 for r in results])),
            rougeL_mean    = float(np.mean([r.rougeL for r in results])),
            bertscore_mean = float(np.mean([r.bertscore_f1 for r in results])),
            wc_delta_mean  = float(np.mean([r.wc_delta for r in results])),
        )


# ── CLI runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/task_a_eval.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    evaluator = TaskAEvaluator()
    results, agg = evaluator.evaluate(n_users=args.n_users)

    output = {
        "aggregate": asdict(agg),
        "per_sample": [asdict(r) for r in results],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*55)
    print("  TASK A EVALUATION RESULTS")
    print("="*55)
    print(f"  Samples:       {agg.n_samples}")
    print(f"  Rating RMSE:   {agg.rmse:.4f}  ← target: < 1.0")
    print(f"  Rating MAE:    {agg.mae:.4f}")
    print(f"  ROUGE-1:       {agg.rouge1_mean:.4f}")
    print(f"  ROUGE-2:       {agg.rouge2_mean:.4f}")
    print(f"  ROUGE-L:       {agg.rougeL_mean:.4f}")
    print(f"  BERTScore F1:  {agg.bertscore_mean:.4f}")
    print(f"  Avg WC delta:  {agg.wc_delta_mean:.1f} words")
    print(f"\n  Full results saved to: {args.output}")