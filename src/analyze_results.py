"""
Phase 13 — Statistical Analysis

Menganalisis hasil LLM-as-a-Judge dan eksperimen AniRAG-v2.

Input:
    results/judge_scores_v22.jsonl
    results/experiment_v1.jsonl

Analisis utama:
    - Validasi pasangan A/B berdasarkan query_id
    - Statistik deskriptif
    - Wilcoxon Signed-Rank Test A vs B
      untuk:
        * relevance
        * factual_accuracy
        * coherence
    - Effect size rank-biserial correlation
    - Direction of paired differences
    - Holm correction untuk multiple testing
    - Refusal-rate analysis dari 100 kasus OOS + adversarial

Tidak ada:
    - Condition C
    - B vs C
    - GPU
    - LLM
    - Internet

Output:
    results/statistical_analysis_v1.json
    docs/STATISTICAL_ANALYSIS.md
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JUDGE_SCORES_PATH = PROJECT_ROOT / "results" / "judge_scores_v22.jsonl"
EXPERIMENT_PATH = PROJECT_ROOT / "results" / "experiment_v1.jsonl"

JSON_OUT_PATH = (
    PROJECT_ROOT / "results" / "statistical_analysis_v1.json"
)

MD_OUT_PATH = (
    PROJECT_ROOT / "docs" / "STATISTICAL_ANALYSIS.md"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_JUDGE_RECORDS = 400
EXPECTED_QUERY_IDS = 200

CONDITIONS = {"A", "B"}

RETRIEVAL_CATEGORIES = {
    "similarity",
    "factual",
    "attribute_filtering",
    "multi_turn_refinement",
}

SAFETY_CATEGORIES = {
    "out_of_scope",
    "adversarial",
}

JUDGE_METRICS = {
    "relevance": "Relevance",
    "factual_accuracy": "Factual Accuracy",
    "coherence": "Coherence",
}

SCORE_MIN = 1
SCORE_MAX = 5

# Explicitly define Wilcoxon handling of zero differences.
#
# "wilcox" removes zero differences before ranking.
# This is appropriate for the paired signed-rank comparison
# because equal A/B scores provide no directional evidence.
WILCOXON_ZERO_METHOD = "wilcox"

ALPHA = 0.05


# ============================================================
# BASIC JSONL LOADING
# ============================================================

def load_jsonl(path: Path):
    """Load JSONL file and return list of records."""

    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON malformed pada {path}, "
                    f"baris {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


# ============================================================
# JUDGE DATA VALIDATION
# ============================================================

def validate_judge_records(records):
    """
    Validate:
        - jumlah record
        - query_id
        - condition
        - duplicate pair
        - score range
        - A/B completeness
        - retrieval categories
    """

    print("=" * 60)
    print("VALIDASI JUDGE SCORES")
    print("=" * 60)

    errors = []

    if len(records) != EXPECTED_JUDGE_RECORDS:
        errors.append(
            f"Expected {EXPECTED_JUDGE_RECORDS} records, "
            f"found {len(records)}."
        )

    pair_counter = Counter()
    query_conditions = defaultdict(set)

    for index, record in enumerate(records, start=1):

        query_id = record.get("query_id")
        condition = record.get("condition")
        category = record.get("category")

        if not query_id:
            errors.append(
                f"Record #{index}: query_id kosong."
            )
            continue

        if condition not in CONDITIONS:
            errors.append(
                f"Record #{index}: condition tidak valid: "
                f"{condition!r}"
            )

        pair_key = (query_id, condition)
        pair_counter[pair_key] += 1
        query_conditions[query_id].add(condition)

        judge = record.get("judge")

        if not isinstance(judge, dict):
            errors.append(
                f"{query_id}/{condition}: field judge tidak valid."
            )
            continue

        for metric in JUDGE_METRICS:

            value = judge.get(metric)

            if not isinstance(value, (int, float)):
                errors.append(
                    f"{query_id}/{condition}: "
                    f"{metric} bukan numeric."
                )
                continue

            if not SCORE_MIN <= value <= SCORE_MAX:
                errors.append(
                    f"{query_id}/{condition}: "
                    f"{metric}={value} di luar range "
                    f"{SCORE_MIN}-{SCORE_MAX}."
                )

        # Judge scores hanya digunakan untuk retrieval-dependent
        # categories.
        if category not in RETRIEVAL_CATEGORIES:
            errors.append(
                f"{query_id}/{condition}: "
                f"kategori {category!r} tidak termasuk "
                f"retrieval categories."
            )

    duplicates = {
        pair: count
        for pair, count in pair_counter.items()
        if count > 1
    }

    if duplicates:
        errors.append(
            f"Ditemukan duplicate pair: {duplicates}"
        )

    unique_query_ids = set(query_conditions.keys())

    if len(unique_query_ids) != EXPECTED_QUERY_IDS:
        errors.append(
            f"Expected {EXPECTED_QUERY_IDS} query IDs, "
            f"found {len(unique_query_ids)}."
        )

    incomplete_pairs = {
        query_id: sorted(conditions)
        for query_id, conditions in query_conditions.items()
        if conditions != CONDITIONS
    }

    if incomplete_pairs:
        errors.append(
            f"A/B tidak lengkap: {incomplete_pairs}"
        )

    if errors:
        print("[FAIL] Validasi judge scores gagal.")

        for error in errors:
            print(f"  - {error}")

        raise ValueError(
            "Judge scores tidak lolos validasi integritas."
        )

    print("[OK] Total records      :", len(records))
    print("[OK] Unique query IDs   :", len(unique_query_ids))
    print("[OK] Condition A        :", sum(
        1 for r in records if r["condition"] == "A"
    ))
    print("[OK] Condition B        :", sum(
        1 for r in records if r["condition"] == "B"
    ))
    print("[OK] Duplicate pairs    :", len(duplicates))
    print("[OK] Complete A/B pairs :", len(unique_query_ids))
    print()


# ============================================================
# BUILD A/B PAIRS
# ============================================================

def build_pairs(records):
    """
    Build paired records using query_id.

    IMPORTANT:
    Pairing is based on query_id, not row order.
    """

    paired = defaultdict(dict)

    for record in records:
        query_id = record["query_id"]
        condition = record["condition"]

        paired[query_id][condition] = record

    # Ensure every query has A and B.
    incomplete = {
        query_id: sorted(data.keys())
        for query_id, data in paired.items()
        if set(data.keys()) != CONDITIONS
    }

    if incomplete:
        raise ValueError(
            f"Pasangan A/B tidak lengkap: {incomplete}"
        )

    return dict(paired)


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def median(values):
    """Return median as float."""

    if len(values) == 0:
        return None

    return float(np.median(values))


def mean(values):
    """Return mean as float."""

    if len(values) == 0:
        return None

    return float(np.mean(values))


def paired_metric_values(pairs, metric):
    """
    Extract paired A/B scores for one metric.
    """

    x = []
    y = []

    for query_id in sorted(pairs):

        judge_a = pairs[query_id]["A"]["judge"]
        judge_b = pairs[query_id]["B"]["judge"]

        x.append(float(judge_a[metric]))
        y.append(float(judge_b[metric]))

    return np.array(x, dtype=float), np.array(y, dtype=float)


def describe_metric(pairs, metric):
    """
    Descriptive statistics for one paired metric.
    """

    x, y = paired_metric_values(pairs, metric)

    differences = y - x

    b_greater = int(np.sum(differences > 0))
    equal = int(np.sum(differences == 0))
    b_less = int(np.sum(differences < 0))

    return {
        "n_pairs": int(len(x)),
        "mean_A": round(mean(x), 4),
        "mean_B": round(mean(y), 4),
        "median_A": round(median(x), 4),
        "median_B": round(median(y), 4),
        "mean_difference_B_minus_A": round(mean(differences), 4),
        "median_difference_B_minus_A": round(
            median(differences), 4
        ),
        "B_greater_than_A": b_greater,
        "B_equal_A": equal,
        "B_less_than_A": b_less,
        "zero_difference_rate": round(
            equal / len(differences), 4
        ),
    }


# ============================================================
# EFFECT SIZE
# ============================================================

def rank_biserial_effect_size(x, y):
    """
    Rank-biserial correlation for paired data.

    Based on non-zero paired differences.

    Interpretation:
        positive -> B tends to score higher than A
        negative -> A tends to score higher than B
        near zero -> little directional difference

    Formula:

        r_rb = (W_positive - W_negative) / W_total

    where W_total is the sum of ranks for all
    non-zero absolute differences.
    """

    differences = np.asarray(y) - np.asarray(x)

    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return 0.0

    absolute_values = np.abs(nonzero)

    # Rank absolute differences using average ranks for ties.
    order = np.argsort(absolute_values)

    ranks = np.empty(len(absolute_values), dtype=float)

    sorted_values = absolute_values[order]

    start = 0

    while start < len(sorted_values):
        end = start + 1

        while (
            end < len(sorted_values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank

        start = end

    positive_rank_sum = float(
        np.sum(ranks[nonzero > 0])
    )

    negative_rank_sum = float(
        np.sum(ranks[nonzero < 0])
    )

    total_rank_sum = positive_rank_sum + negative_rank_sum

    if total_rank_sum == 0:
        return 0.0

    return (
        (positive_rank_sum - negative_rank_sum)
        / total_rank_sum
    )


def interpret_effect_size(effect):
    """
    Descriptive interpretation of rank-biserial correlation.

    Thresholds are used only as conventional magnitude
    descriptors, not as claims about practical importance.
    """

    magnitude = abs(effect)

    if magnitude < 0.1:
        label = "negligible"
    elif magnitude < 0.3:
        label = "small"
    elif magnitude < 0.5:
        label = "moderate"
    else:
        label = "large"

    if effect > 0:
        direction = "B lebih tinggi daripada A"
    elif effect < 0:
        direction = "A lebih tinggi daripada B"
    else:
        direction = "tidak ada arah perbedaan"

    return f"{label}; {direction}"


# ============================================================
# WILCOXON
# ============================================================

def run_wilcoxon_for_metric(pairs, metric):
    """
    Run paired Wilcoxon signed-rank test A vs B.
    """

    x, y = paired_metric_values(pairs, metric)

    differences = y - x

    n_pairs = len(x)
    n_nonzero = int(np.sum(differences != 0))
    n_zero = int(np.sum(differences == 0))

    effect = rank_biserial_effect_size(x, y)

    if n_nonzero == 0:

        return {
            "metric": metric,
            "n_pairs": n_pairs,
            "n_nonzero_pairs": 0,
            "n_zero_difference_pairs": n_zero,
            "wilcoxon_statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "effect_size_rank_biserial": 0.0,
            "effect_interpretation": (
                "negligible; tidak ada perbedaan "
                "berpasangan"
            ),
            "zero_method": WILCOXON_ZERO_METHOD,
        }

    statistic, p_value = wilcoxon(
        x,
        y,
        zero_method=WILCOXON_ZERO_METHOD,
        alternative="two-sided",
        method="auto",
    )

    return {
        "metric": metric,
        "n_pairs": n_pairs,
        "n_nonzero_pairs": n_nonzero,
        "n_zero_difference_pairs": n_zero,
        "wilcoxon_statistic": round(
            float(statistic), 4
        ),
        "p_value": float(p_value),
        "significant_at_0.05": bool(
            p_value < ALPHA
        ),
        "effect_size_rank_biserial": round(
            float(effect), 4
        ),
        "effect_interpretation": interpret_effect_size(
            effect
        ),
        "zero_method": WILCOXON_ZERO_METHOD,
        "alternative": "two-sided",
    }


def run_all_wilcoxon(pairs):
    """
    Run Wilcoxon separately for all three judge metrics,
    together with paired descriptive statistics.
    """

    results = {}

    for metric in JUDGE_METRICS:

        descriptive = describe_metric(
            pairs,
            metric,
        )

        wilcoxon_result = run_wilcoxon_for_metric(
            pairs,
            metric,
        )

        wilcoxon_result.update({
            "mean_A": descriptive["mean_A"],
            "mean_B": descriptive["mean_B"],
            "median_A": descriptive["median_A"],
            "median_B": descriptive["median_B"],
            "mean_difference_B_minus_A": (
                descriptive["mean_difference_B_minus_A"]
            ),
            "median_difference_B_minus_A": (
                descriptive["median_difference_B_minus_A"]
            ),
            "B_greater_than_A": (
                descriptive["B_greater_than_A"]
            ),
            "B_equal_A": (
                descriptive["B_equal_A"]
            ),
            "B_less_than_A": (
                descriptive["B_less_than_A"]
            ),
            "zero_difference_rate": (
                descriptive["zero_difference_rate"]
            ),
        })

        results[metric] = wilcoxon_result

    return results


# ============================================================
# HOLM MULTIPLE-TESTING CORRECTION
# ============================================================

def holm_correction(wilcoxon_results):
    """
    Holm-Bonferroni correction for the three primary
    Wilcoxon tests.

    Returns adjusted p-values and adjusted significance.
    """

    entries = []

    for metric, result in wilcoxon_results.items():

        entries.append(
            (
                metric,
                float(result["p_value"]),
            )
        )

    entries.sort(key=lambda item: item[1])

    m = len(entries)

    adjusted = {}

    previous_adjusted = 0.0

    for rank, (metric, p_value) in enumerate(
        entries,
        start=1,
    ):

        adjusted_p = min(
            1.0,
            (m - rank + 1) * p_value,
        )

        # Holm adjusted p-values must be monotonic.
        adjusted_p = max(
            adjusted_p,
            previous_adjusted,
        )

        previous_adjusted = adjusted_p

        adjusted[metric] = float(adjusted_p)

    for metric, adjusted_p in adjusted.items():

        wilcoxon_results[metric][
            "holm_adjusted_p_value"
        ] = adjusted_p

        wilcoxon_results[metric][
            "significant_after_holm_0.05"
        ] = bool(adjusted_p < ALPHA)

    return wilcoxon_results


# ============================================================
# REFUSAL ANALYSIS
# ============================================================

def analyze_refusal_rate(experiment_records):
    """
    Analyze guardrail/refusal behavior from experiment_v1.jsonl.

    This analysis is intentionally separate from the main
    Wilcoxon comparison.

    Categories:
        - out_of_scope
        - adversarial
    """

    selected = [
        record
        for record in experiment_records
        if record.get("category") in SAFETY_CATEGORIES
    ]

    by_category_condition = defaultdict(list)

    for record in selected:

        category = record["category"]
        condition = record["condition"]

        by_category_condition[
            (category, condition)
        ].append(record)

    result = {}

    for category in sorted(SAFETY_CATEGORIES):

        result[category] = {}

        for condition in sorted(CONDITIONS):

            records = by_category_condition.get(
                (category, condition),
                [],
            )

            total = len(records)

            blocked = sum(
                1
                for record in records
                if record.get("blocked") is True
            )

            refusal_present = sum(
                1
                for record in records
                if record.get("refusal")
            )

            response_count = sum(
                1
                for record in records
                if isinstance(
                    record.get("response"),
                    str,
                )
                and record.get("response").strip()
            )

            result[category][condition] = {
                "n_records": total,
                "blocked_count": blocked,
                "blocked_rate": round(
                    blocked / total,
                    4,
                ) if total else None,
                "refusal_field_present_count": (
                    refusal_present
                ),
                "non_empty_response_count": (
                    response_count
                ),
            }

    # Aggregate across OOS + adversarial.
    result["overall"] = {}

    for condition in sorted(CONDITIONS):

        records = [
            record
            for record in selected
            if record.get("condition") == condition
        ]

        total = len(records)

        blocked = sum(
            1
            for record in records
            if record.get("blocked") is True
        )

        result["overall"][condition] = {
            "n_records": total,
            "blocked_count": blocked,
            "blocked_rate": round(
                blocked / total,
                4,
            ) if total else None,
        }

    return result


# ============================================================
# CATEGORY SUMMARY
# ============================================================

def category_summary(pairs):
    """
    Descriptive mean per category and condition.
    """

    output = {}

    for category in sorted(RETRIEVAL_CATEGORIES):

        output[category] = {}

        category_pairs = {
            query_id: pair
            for query_id, pair in pairs.items()
            if pair["A"].get("category") == category
        }

        for condition in sorted(CONDITIONS):

            records = [
                pair[condition]
                for pair in category_pairs.values()
            ]

            metric_means = {}

            for metric in JUDGE_METRICS:
                values = [
                    record["judge"][metric]
                    for record in records
                ]

                metric_means[metric] = round(
                    float(np.mean(values)),
                    4,
                )

            output[category][condition] = {
                "n_query": len(records),
                **metric_means,
            }

    return output

# ============================================================
# REPORT GENERATION
# ============================================================

def build_report(
    judge_records,
    pairs,
    wilcoxon_results,
    category_stats,
    refusal_results,
):
    return {
        "phase": 13,
        "title": "Statistical Analysis — AniRAG-v2",
        "status": "complete",
        "design": {
            "comparison": "A_vs_B",
            "condition_A": (
                "SLM-only without retrieval"
            ),
            "condition_B": (
                "SLM + RAG"
            ),
            "paired_unit": "query_id",
            "n_query_ids": len(pairs),
            "n_judge_records": len(judge_records),
        },
        "input": {
            "judge_scores": str(
                JUDGE_SCORES_PATH.relative_to(PROJECT_ROOT)
            ),
            "experiment_results": str(
                EXPERIMENT_PATH.relative_to(PROJECT_ROOT)
            ),
        },
        "judge_integrity": {
            "total_records": len(judge_records),
            "unique_query_ids": len(pairs),
            "expected_records": EXPECTED_JUDGE_RECORDS,
            "expected_query_ids": EXPECTED_QUERY_IDS,
            "conditions": sorted(CONDITIONS),
            "pairing_method": "query_id",
        },
        "category_statistics": category_stats,
        "wilcoxon_signed_rank": wilcoxon_results,
        "refusal_rate_analysis": refusal_results,
        "methodology": {
            "alpha": ALPHA,
            "alternative": "two-sided",
            "zero_method": WILCOXON_ZERO_METHOD,
            "multiple_testing_correction": (
                "Holm-Bonferroni"
            ),
            "effect_size": (
                "rank-biserial correlation"
            ),
        },
        "notes": [
            (
                "Wilcoxon digunakan untuk paired comparison "
                "A vs B berdasarkan query_id."
            ),
            (
                "Condition C tidak termasuk dalam eksperimen "
                "final dan tidak dianalisis."
            ),
            (
                "100 kasus out_of_scope dan adversarial "
                "digunakan untuk refusal/robustness analysis "
                "terpisah, bukan primary Wilcoxon comparison."
            ),
        ],
    }


def save_json(report):
    """Save JSON statistical report."""

    JSON_OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSON_OUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

def format_p_value(p_value):
    """
    Format p-value untuk laporan Markdown.

    p < 0.001 ditampilkan sebagai '< 0.001'
    agar tidak menghasilkan 'p = 0.000000'.
    """

    p_value = float(p_value)

    if p_value < 0.001:
        return "< 0.001"

    return f"{p_value:.6f}"


def save_markdown(report):
    """Generate human-readable Markdown report."""

    lines = []

    lines.append("# Phase 13 — Statistical Analysis")
    lines.append("")
    lines.append(
        "Statistical analysis hasil evaluasi AniRAG-v2 "
        "dengan paired comparison Condition A vs Condition B."
    )
    lines.append("")

    lines.append("## 1. Experimental Design")
    lines.append("")
    lines.append(
        "- Condition A: SLM-only without retrieval"
    )
    lines.append(
        "- Condition B: SLM + RAG"
    )
    lines.append(
        "- Paired unit: `query_id`"
    )
    lines.append(
        f"- Query IDs: {report['design']['n_query_ids']}"
    )
    lines.append(
        f"- Judge records: {report['design']['n_judge_records']}"
    )
    lines.append("")

    lines.append("## 2. Category Statistics")
    lines.append("")
    lines.append(
        "| Category | Condition | N | Relevance | "
        "Factual Accuracy | Coherence |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for category, conditions in report[
        "category_statistics"
    ].items():

        for condition, values in conditions.items():

            lines.append(
                f"| {category} | {condition} | "
                f"{values['n_query']} | "
                f"{values['relevance']:.4f} | "
                f"{values['factual_accuracy']:.4f} | "
                f"{values['coherence']:.4f} |"
            )

    lines.append("")

    lines.append("## 3. Wilcoxon Signed-Rank Test")
    lines.append("")
    lines.append(
        "Perbandingan dilakukan secara berpasangan "
        "antara Condition A dan B berdasarkan `query_id`."
    )
    lines.append("")

    lines.append(
        "| Metric | N | A Mean | B Mean | "
        "Median Δ(B-A) | W | p | Holm p | Effect |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for metric, result in report[
        "wilcoxon_signed_rank"
    ].items():

        lines.append(
            f"| {metric} | "
            f"{result['n_pairs']} | "
            f"{result.get('mean_A', '-') if 'mean_A' in result else '-'} | "
            f"{result.get('mean_B', '-') if 'mean_B' in result else '-'} | "
            f"{result.get('median_difference_B_minus_A', '-') if 'median_difference_B_minus_A' in result else '-'} | "
            f"{result['wilcoxon_statistic']:.4f} | "
            f"{format_p_value(result['p_value'])} | "
            f"{format_p_value(result['holm_adjusted_p_value'])} | "
            f"{result['effect_size_rank_biserial']:.4f} |"
        )

    lines.append("")

    lines.append("### Statistical Method")
    lines.append("")
    lines.append(
        f"- Test: Wilcoxon Signed-Rank Test"
    )
    lines.append(
        f"- Comparison: A vs B"
    )
    lines.append(
        f"- Alternative: two-sided"
    )
    lines.append(
        f"- Zero method: `{WILCOXON_ZERO_METHOD}`"
    )
    lines.append(
        f"- Alpha: {ALPHA}"
    )
    lines.append(
        "- Multiple-testing correction: Holm-Bonferroni"
    )
    lines.append(
        "- Effect size: rank-biserial correlation"
    )
    lines.append("")

    lines.append("## 4. Refusal / Safety Analysis")
    lines.append("")
    lines.append(
        "Analisis ini menggunakan `experiment_v1.jsonl` "
        "dan tidak dimasukkan ke primary Wilcoxon comparison."
    )
    lines.append("")

    lines.append(
        "| Category | Condition | N | Blocked | Blocked Rate |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    refusal = report["refusal_rate_analysis"]

    for category in SAFETY_CATEGORIES:

        for condition in sorted(CONDITIONS):

            values = refusal[category][condition]

            lines.append(
                f"| {category} | {condition} | "
                f"{values['n_records']} | "
                f"{values['blocked_count']} | "
                f"{values['blocked_rate']} |"
            )

    lines.append("")

    lines.append("## 5. Methodological Notes")
    lines.append("")
    lines.append(
        "- Pairing dilakukan berdasarkan `query_id`, bukan urutan baris."
    )
    lines.append(
        "- Condition C tidak dianalisis."
    )
    lines.append(
        "- OOS dan adversarial tidak dimasukkan ke Wilcoxon utama."
    )
    lines.append(
        "- Coherence perlu diinterpretasikan dengan memperhatikan "
        "zero-difference/ceiling effect."
    )
    lines.append("")

    lines.append("## 6. Phase Status")
    lines.append("")
    lines.append(
        "**Phase 13: COMPLETE / FROZEN setelah hasil "
        "diverifikasi secara manual.**"
    )
    lines.append("")

    MD_OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MD_OUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHASE 13 — STATISTICAL ANALYSIS")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # 1. Load judge scores
    # --------------------------------------------------------

    print("[1/6] Membaca judge_scores_v22.jsonl...")

    judge_records = load_jsonl(
        JUDGE_SCORES_PATH
    )

    validate_judge_records(
        judge_records
    )

    # --------------------------------------------------------
    # 2. Build A/B pairs
    # --------------------------------------------------------

    print("[2/6] Membentuk pasangan A/B berdasarkan query_id...")

    pairs = build_pairs(
        judge_records
    )

    print(
        f"[OK] {len(pairs)} pasangan A/B terbentuk."
    )
    print()

    # --------------------------------------------------------
    # 3. Category statistics
    # --------------------------------------------------------

    print("[3/6] Menghitung statistik deskriptif...")

    category_stats = category_summary(
        pairs
    )

    print("[OK] Statistik deskriptif selesai.")
    print()

    # --------------------------------------------------------
    # 4. Wilcoxon
    # --------------------------------------------------------

    print("[4/6] Menjalankan Wilcoxon Signed-Rank A vs B...")

    wilcoxon_results = run_all_wilcoxon(
        pairs
    )

    wilcoxon_results = holm_correction(
        wilcoxon_results
    )

    for metric, result in wilcoxon_results.items():

        print(
            f"  {metric}: "
            f"W={result['wilcoxon_statistic']}, "
            f"p={result['p_value']}, "
            f"Holm p={result['holm_adjusted_p_value']}, "
            f"effect={result['effect_size_rank_biserial']}"
        )

    print()

    # --------------------------------------------------------
    # 5. Refusal analysis
    # --------------------------------------------------------

    print(
        "[5/6] Menganalisis refusal/guardrail "
        "dari experiment_v1.jsonl..."
    )

    experiment_records = load_jsonl(
        EXPERIMENT_PATH
    )

    refusal_results = analyze_refusal_rate(
        experiment_records
    )

    print("[OK] Refusal analysis selesai.")
    print()

    # --------------------------------------------------------
    # 6. Save reports
    # --------------------------------------------------------

    print("[6/6] Menyimpan report...")

    report = build_report(
        judge_records=judge_records,
        pairs=pairs,
        wilcoxon_results=wilcoxon_results,
        category_stats=category_stats,
        refusal_results=refusal_results,
    )

    save_json(report)
    save_markdown(report)

    print()
    print("=" * 60)
    print("PHASE 13 — SELESAI")
    print("=" * 60)
    print(
        f"JSON   : {JSON_OUT_PATH}"
    )
    print(
        f"MD     : {MD_OUT_PATH}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()