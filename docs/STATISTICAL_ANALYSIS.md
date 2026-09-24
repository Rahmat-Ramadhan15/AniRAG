# Phase 13 — Statistical Analysis

Statistical analysis hasil evaluasi AniRAG-v2 dengan paired comparison Condition A vs Condition B.

## 1. Experimental Design

- Condition A: SLM-only without retrieval
- Condition B: SLM + RAG
- Paired unit: `query_id`
- Query IDs: 200
- Judge records: 400

## 2. Category Statistics

| Category | Condition | N | Relevance | Factual Accuracy | Coherence |
|---|---:|---:|---:|---:|---:|
| attribute_filtering | A | 50 | 2.9200 | 2.6800 | 4.4000 |
| attribute_filtering | B | 50 | 3.5200 | 3.3200 | 4.5800 |
| factual | A | 50 | 1.7800 | 3.6400 | 4.9800 |
| factual | B | 50 | 4.5000 | 4.6600 | 5.0000 |
| multi_turn_refinement | A | 50 | 1.9000 | 3.0200 | 4.5200 |
| multi_turn_refinement | B | 50 | 3.2400 | 3.6600 | 4.4400 |
| similarity | A | 50 | 3.2600 | 3.0600 | 4.7800 |
| similarity | B | 50 | 2.8200 | 3.4800 | 4.5200 |

## 3. Wilcoxon Signed-Rank Test

Perbandingan dilakukan secara berpasangan antara Condition A dan B berdasarkan `query_id`.

| Metric | N | A Mean | B Mean | Median Δ(B-A) | W | p | Holm p | Effect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relevance | 200 | 2.465 | 3.52 | 1.0 | 2044.5000 | < 0.001 | < 0.001 | 0.6190 |
| factual_accuracy | 200 | 3.1 | 3.78 | 0.0 | 2184.0000 | < 0.001 | < 0.001 | 0.4709 |
| coherence | 200 | 4.67 | 4.635 | 0.0 | 804.5000 | 0.531425 | 0.531425 | -0.0910 |

### Statistical Method

- Test: Wilcoxon Signed-Rank Test
- Comparison: A vs B
- Alternative: two-sided
- Zero method: `wilcox`
- Alpha: 0.05
- Multiple-testing correction: Holm-Bonferroni
- Effect size: rank-biserial correlation

## 4. Refusal / Safety Analysis

Analisis ini menggunakan `experiment_v1.jsonl` dan tidak dimasukkan ke primary Wilcoxon comparison.

| Category | Condition | N | Blocked | Blocked Rate |
|---|---:|---:|---:|---:|
| adversarial | A | 50 | 8 | 0.16 |
| adversarial | B | 50 | 8 | 0.16 |
| out_of_scope | A | 50 | 8 | 0.16 |
| out_of_scope | B | 50 | 8 | 0.16 |

## 5. Methodological Notes

- Pairing dilakukan berdasarkan `query_id`, bukan urutan baris.
- Condition C tidak dianalisis.
- OOS dan adversarial tidak dimasukkan ke Wilcoxon utama.
- Coherence perlu diinterpretasikan dengan memperhatikan zero-difference/ceiling effect.

## 6. Phase Status

**Phase 13: COMPLETE / FROZEN setelah hasil diverifikasi secara manual.**
