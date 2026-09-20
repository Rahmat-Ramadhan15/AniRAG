"""
AniRAG-v2 — Phase 9: Retrieval Evaluation
==========================================

Evaluates the FINAL retrieval path used by AniRAG-v2.

Important:
- Uses RagPipeline.retrieve_filtered(), not direct FAISS search.
- For multi-turn, uses RagPipeline.build_effective_query() with exactly
  the previous user turn + current follow-up.
- Does NOT load the SLM/LLM.
- Does NOT evaluate generation quality.
- Uses tests/test_set_v1.jsonl as the fixed evaluation set.

Primary metrics:
A. Similarity
   Precision@3, Precision@5, Precision@10

B. Factual
   Recall@3, Recall@5, Recall@10, MRR

C. Attribute Filtering
   Precision@3, Precision@5, Precision@10
   Primary focus: Precision@5

D. Multi-turn Refinement
   Precision@3, Precision@5, Precision@10

E/F are not scored as retrieval metrics here:
- Out-of-Scope requires response-level refusal evaluation.
- Adversarial requires response-level robustness / hallucination evaluation.
Those belong to the generation/guardrail evaluation stages.

Output:
    results/retrieval_v1.json
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SET_PATH = PROJECT_ROOT / "tests" / "test_set_v1.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "retrieval_v1.json"

# The evaluation is fixed at these cutoffs.
K_VALUES = (3, 5, 10)

# Only these categories have retrieval ground truth + retrieval metrics.
RETRIEVAL_CATEGORIES = (
    "similarity",
    "factual",
    "attribute_filtering",
    "multi_turn_refinement",
)

EXPECTED_COUNTS = {
    "similarity": 50,
    "factual": 50,
    "attribute_filtering": 50,
    "multi_turn_refinement": 50,
}

# ---------------------------------------------------------------------------
# Import RagPipeline from src/
# ---------------------------------------------------------------------------

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pipeline import RagPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Test-set loading / validation
# ---------------------------------------------------------------------------

def load_test_set() -> list[dict[str, Any]]:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"Test set tidak ditemukan: {TEST_SET_PATH}"
        )

    rows: list[dict[str, Any]] = []

    with TEST_SET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise TypeError(
                    f"Test case pada baris {line_number} bukan object."
                )

            rows.append(item)

    return rows


def validate_test_set(test_set: list[dict[str, Any]]) -> None:
    if len(test_set) != 300:
        raise ValueError(
            f"Test set harus berisi 300 kasus. Aktual: {len(test_set)}"
        )

    counts: dict[str, int] = {}

    for item in test_set:
        category = item.get("category")
        counts[category] = counts.get(category, 0) + 1

    expected_all = {
        "similarity": 50,
        "factual": 50,
        "attribute_filtering": 50,
        "multi_turn_refinement": 50,
        "out_of_scope": 50,
        "adversarial": 50,
    }

    if counts != expected_all:
        raise ValueError(
            f"Distribusi kategori test set tidak sesuai.\n"
            f"Expected: {expected_all}\n"
            f"Actual:   {counts}"
        )

    ids = [item.get("id") for item in test_set]

    if len(ids) != len(set(ids)):
        raise ValueError("Terdapat duplicate test ID.")

    for item in test_set:
        category = item.get("category")

        if category not in RETRIEVAL_CATEGORIES:
            continue

        gt = item.get("ground_truth_mal_ids")

        if not isinstance(gt, list) or not gt:
            raise ValueError(
                f"{item.get('id')}: ground_truth_mal_ids kosong/tidak valid."
            )

        if category == "multi_turn_refinement":
            query = item.get("query")

            if not isinstance(query, list) or len(query) != 2:
                raise ValueError(
                    f"{item.get('id')}: multi-turn harus tepat 2 turn."
                )

            if not all(isinstance(turn, str) and turn.strip() for turn in query):
                raise ValueError(
                    f"{item.get('id')}: setiap multi-turn harus berupa string "
                    "non-empty."
                )

        else:
            query = item.get("query")

            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"{item.get('id')}: query harus berupa string non-empty."
                )


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_ids: list[int],
    ground_truth_ids: set[int],
    k: int,
) -> float:
    """
    Precision@k = relevant retrieved items / k.

    The denominator remains k so that all test cases are evaluated against
    the same cutoff. If the pipeline returns fewer than k documents, this
    naturally penalizes incomplete retrieval.
    """
    if k <= 0:
        raise ValueError("k harus > 0.")

    top_k = retrieved_ids[:k]
    hits = sum(1 for mal_id in top_k if mal_id in ground_truth_ids)

    return hits / k


def recall_at_k(
    retrieved_ids: list[int],
    ground_truth_ids: set[int],
    k: int,
) -> float:
    """Recall@k = relevant retrieved items / total relevant items."""
    if not ground_truth_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(ground_truth_ids))

    return hits / len(ground_truth_ids)


def reciprocal_rank(
    retrieved_ids: list[int],
    ground_truth_ids: set[int],
) -> float:
    """
    Reciprocal rank of the first relevant result.

    Returns 0.0 when no relevant result is retrieved.
    """
    for rank, mal_id in enumerate(retrieved_ids, start=1):
        if mal_id in ground_truth_ids:
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def run_single_turn(
    pipeline: RagPipeline,
    query: str,
    top_k: int,
) -> dict[str, Any]:
    """
    Run the same final retrieval path used by the application.

    We intentionally call retrieve_filtered(), which includes:
      attribute detection
      anchor detection
      semantic retrieval
      hard filtering
      franchise exclusion
      reranking
      final top-k
    """
    result = pipeline.retrieve_filtered(
        query,
        top_k=top_k,
    )

    results = result.get("results", [])

    retrieved_ids: list[int] = []

    for item in results:
        try:
            retrieved_ids.append(int(item["mal_id"]))
        except (KeyError, TypeError, ValueError):
            continue

    return {
        "effective_query": query,
        "retrieved_ids": retrieved_ids,
        "retrieval_mode": result.get("retrieval_mode"),
        "filters": result.get("filters"),
        "candidate_k": result.get("candidate_k"),
        "before_filtering": result.get("candidate_count"),
        "after_filtering": result.get("after_attribute_filter"),
        "after_anchor_exclusion": result.get("after_anchor_exclusion"),
        "raw_result_count": len(results),
    }


def run_multi_turn(
    pipeline: RagPipeline,
    turns: list[str],
    top_k: int,
) -> dict[str, Any]:
    """
    Evaluate retrieval after the pipeline's official 2-turn contextual
    integration.

    Turn 1 becomes the immediately preceding user message in history.
    Turn 2 is passed as the current query.

    This deliberately does NOT implement cumulative 3+ turn memory.
    """
    turn_1, turn_2 = turns

    history = [
        {
            "role": "user",
            "content": turn_1,
        }
    ]

    effective_query = pipeline.build_effective_query(
        turn_2,
        history=history,
    )

    result = pipeline.retrieve_filtered(
        effective_query,
        top_k=top_k,
    )

    results = result.get("results", [])

    retrieved_ids: list[int] = []

    for item in results:
        try:
            retrieved_ids.append(int(item["mal_id"]))
        except (KeyError, TypeError, ValueError):
            continue

    return {
        "effective_query": effective_query,
        "retrieved_ids": retrieved_ids,
        "retrieval_mode": result.get("retrieval_mode"),
        "filters": result.get("filters"),
        "candidate_k": result.get("candidate_k"),
        "before_filtering": result.get("candidate_count"),
        "after_filtering": result.get("after_attribute_filter"),
        "after_anchor_exclusion": result.get("after_anchor_exclusion"),
        "raw_result_count": len(results),
    }


# ---------------------------------------------------------------------------
# Case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    pipeline: RagPipeline,
    item: dict[str, Any],
) -> dict[str, Any]:
    category = item["category"]
    test_id = item["id"]
    ground_truth_ids = {
        int(mal_id)
        for mal_id in item["ground_truth_mal_ids"]
    }

    started = time.perf_counter()

    # Suppress verbose diagnostic prints from RagPipeline for a clean
    # evaluation log. The detailed metrics are still stored in JSON.
    captured_stdout = io.StringIO()

    with contextlib.redirect_stdout(captured_stdout):
        if category == "multi_turn_refinement":
            retrieval = run_multi_turn(
                pipeline,
                item["query"],
                top_k=max(K_VALUES),
            )
        else:
            retrieval = run_single_turn(
                pipeline,
                item["query"],
                top_k=max(K_VALUES),
            )

    elapsed = time.perf_counter() - started

    retrieved_ids = retrieval["retrieved_ids"]

    metrics: dict[str, float] = {}

    if category in {
        "similarity",
        "attribute_filtering",
        "multi_turn_refinement",
    }:
        for k in K_VALUES:
            metrics[f"precision@{k}"] = precision_at_k(
                retrieved_ids,
                ground_truth_ids,
                k,
            )

    if category == "factual":
        for k in K_VALUES:
            metrics[f"recall@{k}"] = recall_at_k(
                retrieved_ids,
                ground_truth_ids,
                k,
            )

        metrics["mrr"] = reciprocal_rank(
            retrieved_ids,
            ground_truth_ids,
        )

    result = {
        "id": test_id,
        "category": category,
        "query": item["query"],
        "ground_truth_count": len(ground_truth_ids),
        "ground_truth_mal_ids": sorted(ground_truth_ids),
        "retrieved_mal_ids": retrieved_ids,
        "retrieved_count": len(retrieved_ids),
        "metrics": metrics,
        "retrieval_diagnostics": {
            "retrieval_mode": retrieval.get("retrieval_mode"),
            "filters": retrieval.get("filters"),
            "candidate_k": retrieval.get("candidate_k"),
            "before_filtering": retrieval.get("before_filtering"),
            "after_filtering": retrieval.get("after_filtering"),
            "after_anchor_exclusion": retrieval.get(
                "after_anchor_exclusion"
            ),
            "raw_result_count": retrieval.get("raw_result_count"),
        },
        "effective_query": retrieval.get("effective_query"),
        "latency_seconds": round(elapsed, 6),
    }

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_category(
    case_results: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    subset = [
        result
        for result in case_results
        if result["category"] == category
    ]

    if not subset:
        return {
            "count": 0,
            "metrics": {},
        }

    metric_names: set[str] = set()

    for result in subset:
        metric_names.update(result["metrics"].keys())

    aggregated_metrics: dict[str, float] = {}

    for metric_name in sorted(metric_names):
        values = [
            result["metrics"][metric_name]
            for result in subset
            if metric_name in result["metrics"]
        ]

        aggregated_metrics[metric_name] = (
            mean(values) if values else 0.0
        )

    summary: dict[str, Any] = {
        "count": len(subset),
        "metrics": {
            key: round(value, 6)
            for key, value in aggregated_metrics.items()
        },
    }

    if category == "attribute_filtering":
        summary["primary_metric"] = "precision@5"
        summary["primary_value"] = round(
            aggregated_metrics.get("precision@5", 0.0),
            6,
        )

    return summary


def aggregate_overall(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Overall summary is descriptive only.

    Metrics are NOT pooled across categories because the categories have
    different ground-truth semantics and different primary metrics.
    """
    return {
        "retrieval_cases": len(case_results),
        "categories": {
            category: aggregate_category(case_results, category)
            for category in RETRIEVAL_CATEGORIES
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("AniRAG-v2 — PHASE 9 RETRIEVAL EVALUATION")
    print("=" * 72)

    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Test set     : {TEST_SET_PATH}")
    print(f"[INFO] Output       : {OUTPUT_PATH}")
    print(f"[INFO] K values     : {K_VALUES}")
    print()

    # --------------------------------------------------------------
    # 1. Load + validate test set
    # --------------------------------------------------------------

    print("[1/4] Memuat dan memvalidasi test set...")

    test_set = load_test_set()
    validate_test_set(test_set)

    print("[OK] Test set valid: 300 kasus")
    print("[OK] Retrieval cases: 200 kasus (A-D)")
    print("[OK] OOS + adversarial: 100 kasus tidak diberi retrieval score")
    print()

    # --------------------------------------------------------------
    # 2. Initialize pipeline
    # --------------------------------------------------------------

    print("[2/4] Memuat RagPipeline...")

    init_stdout = io.StringIO()

    with contextlib.redirect_stdout(init_stdout):
        pipeline = RagPipeline()

    print("[OK] RagPipeline siap")
    print(
        f"[INFO] Documents : {len(pipeline.documents)}"
    )
    print(
        f"[INFO] FAISS     : {pipeline.index.ntotal} vectors"
    )
    print(
        f"[INFO] Embedding : {pipeline.embedding_model_name}"
    )
    print(
        f"[INFO] Dimension : {pipeline.embedding_dimension}"
    )
    print()

    # --------------------------------------------------------------
    # 3. Evaluate A-D
    # --------------------------------------------------------------

    retrieval_cases = [
        item
        for item in test_set
        if item["category"] in RETRIEVAL_CATEGORIES
    ]

    print("[3/4] Menjalankan retrieval evaluation...")
    print(
        f"[INFO] Total retrieval cases: {len(retrieval_cases)}"
    )
    print()

    case_results: list[dict[str, Any]] = []

    category_completed: dict[str, int] = {
        category: 0
        for category in RETRIEVAL_CATEGORIES
    }

    errors: list[dict[str, str]] = []

    total_started = time.perf_counter()

    for index, item in enumerate(retrieval_cases, start=1):
        category = item["category"]
        test_id = item["id"]

        try:
            result = evaluate_case(
                pipeline,
                item,
            )
            case_results.append(result)

            category_completed[category] += 1

            print(
                f"[{index:03d}/{len(retrieval_cases)}] "
                f"{test_id:<10} "
                f"{category:<24} "
                "OK"
            )

        except Exception as exc:
            errors.append(
                {
                    "id": test_id,
                    "category": category,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            print(
                f"[{index:03d}/{len(retrieval_cases)}] "
                f"{test_id:<10} "
                f"{category:<24} "
                f"ERROR: {type(exc).__name__}: {exc}"
            )

    total_elapsed = time.perf_counter() - total_started

    # --------------------------------------------------------------
    # 4. Aggregate + save
    # --------------------------------------------------------------

    print()
    print("[4/4] Mengagregasi hasil...")

    category_summary = {
        category: aggregate_category(
            case_results,
            category,
        )
        for category in RETRIEVAL_CATEGORIES
    }

    output = {
        "evaluation": {
            "phase": 9,
            "name": "retrieval_evaluation",
            "test_set": str(TEST_SET_PATH.relative_to(PROJECT_ROOT)),
            "test_set_size": len(test_set),
            "retrieval_case_count": len(retrieval_cases),
            "k_values": list(K_VALUES),
            "retrieval_categories": list(RETRIEVAL_CATEGORIES),
            "pipeline_entrypoint": "RagPipeline.retrieve_filtered",
            "multi_turn_entrypoint": (
                "RagPipeline.build_effective_query"
            ),
            "multi_turn_scope": "exactly_2_turns",
            "llm_used": False,
        },
        "system": {
            "documents": len(pipeline.documents),
            "faiss_vectors": int(pipeline.index.ntotal),
            "faiss_dimension": int(pipeline.index.d),
            "embedding_model": pipeline.embedding_model_name,
            "embedding_dimension": pipeline.embedding_dimension,
            "retrieval_top_k_final": int(pipeline.retrieval_top_k),
            "retrieval_candidate_k": int(
                pipeline.retrieval_candidate_k
            ),
            "semantic_weight": float(pipeline.semantic_weight),
            "score_weight": float(pipeline.score_weight),
        },
        "summary": {
            "completed_cases": len(case_results),
            "failed_cases": len(errors),
            "elapsed_seconds": round(total_elapsed, 3),
            "categories": category_summary,
        },
        "category_completion": category_completed,
        "errors": errors,
        "cases": case_results,
        "deferred_categories": {
            "out_of_scope": {
                "count": 50,
                "reason": (
                    "Requires response-level refusal evaluation; "
                    "not a retrieval metric."
                ),
            },
            "adversarial": {
                "count": 50,
                "reason": (
                    "Requires response-level robustness, "
                    "refusal, and unsupported-claim evaluation."
                ),
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("[OK] Hasil disimpan.")
    print(f"[OK] Output: {OUTPUT_PATH}")
    print()

    print("=" * 72)
    print("RINGKASAN PHASE 9")
    print("=" * 72)

    for category in RETRIEVAL_CATEGORIES:
        summary = category_summary[category]

        print(
            f"\n[{category}] "
            f"n={summary['count']}"
        )

        for metric, value in summary["metrics"].items():
            print(f"  {metric:<15}: {value:.4f}")

        if "primary_metric" in summary:
            print(
                f"  PRIMARY {summary['primary_metric']}: "
                f"{summary['primary_value']:.4f}"
            )

    print()
    print(
        f"[INFO] Completed: {len(case_results)}/"
        f"{len(retrieval_cases)}"
    )
    print(f"[INFO] Errors   : {len(errors)}")
    print(f"[INFO] Time     : {total_elapsed:.2f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
