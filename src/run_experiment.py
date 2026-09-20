"""
AniRAG-v2
PHASE 10 — ABLATION STUDY
=========================

Membandingkan dua kondisi generation:

CONDITION A — SLM ONLY
    use_retrieval=False
    use_enrichment=False

CONDITION B — SLM + RAG
    use_retrieval=True
    use_enrichment=False

Condition C — SLM + RAG + enrichment
    Dikeluarkan dari eksperimen Phase 10.
    Enrichment/Jikan tidak digunakan pada pipeline final AniRAG-v2.

TUJUAN
------
Mengukur perbedaan kualitas respons antara:
    A) SLM tanpa retrieval
    B) SLM dengan retrieval

Kedua kondisi menggunakan:
    - model SLM yang sama
    - tokenizer/chat template yang sama
    - max_new_tokens yang sama
    - temperature yang sama
    - query/test case yang sama

Perbedaan utama:
    - Condition A tidak memperoleh context hasil retrieval.
    - Condition B memperoleh context hasil retrieval.

CATATAN
-------
1. Guardrail dijalankan oleh generate() sebelum percabangan retrieval.
   Query yang diblokir tidak mencapai retrieval maupun LLM.

2. Multi-turn menggunakan history asli:
       turn_1 -> history
       turn_2 -> generate()
   sehingga build_effective_query() dijalankan melalui jalur runtime
   yang sama dengan aplikasi.

3. Runner ini belum melakukan filtering OOS/adversarial.
   Pemilihan test case Phase 10 dilakukan setelah audit guardrail
   terhadap kategori tersebut.

4. Gunakan temperature=0.0 untuk eksperimen agar decoding deterministik.
   Nilai ini merupakan parameter eksperimen dan tidak mengubah
   konfigurasi production secara otomatis.

5. Output:
       tests/experiment_results.jsonl

   Satu baris untuk setiap:
       query x condition

6. Runner ini harus dijalankan pada GPU (misalnya Kaggle)
   karena load_llm() membutuhkan CUDA.
"""

import json
import time
from pathlib import Path

from rag_pipeline import RagPipeline


# ==============================================================
# PATH
# ==============================================================

TEST_SET_PATH = Path("tests/test_set_v1.jsonl")
OUT_PATH = Path("results/experiment_v1.jsonl")


# ==============================================================
# EXPERIMENT CONFIGURATION
# ==============================================================

CONDITIONS = [
    (
        "A",
        {
            "use_retrieval": False,
            "use_enrichment": False,
        },
    ),
    (
        "B",
        {
            "use_retrieval": True,
            "use_enrichment": False,
        },
    ),
]

# Parameter generation dikunci untuk A dan B.
# Temperature 0 -> deterministic decoding.
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0


# ==============================================================
# TEST SET
# ==============================================================

def load_test_set():
    """
    Load test set JSONL.

    Setiap baris harus memiliki minimal:
        id
        category
        query

    Untuk multi-turn:
        query = [turn_1, turn_2]
    """

    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"Test set tidak ditemukan: {TEST_SET_PATH}"
        )

    queries = []

    with open(
        TEST_SET_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris "
                    f"{line_number}: {exc}"
                ) from exc

            if "id" not in item:
                raise ValueError(
                    f"Test case pada baris {line_number} "
                    f"tidak memiliki field 'id'."
                )

            if "category" not in item:
                raise ValueError(
                    f"Test case {item['id']} tidak memiliki "
                    f"field 'category'."
                )

            if "query" not in item:
                raise ValueError(
                    f"Test case {item['id']} tidak memiliki "
                    f"field 'query'."
                )

            queries.append(item)

    if not queries:
        raise ValueError(
            f"Test set kosong: {TEST_SET_PATH}"
        )

    return queries


# ==============================================================
# QUERY PREPARATION
# ==============================================================

def prepare_query(item):
    """
    Menyiapkan query sesuai struktur test set.

    Single-turn:
        query = "..."

    Multi-turn:
        query = ["turn 1", "turn 2"]

    Untuk multi-turn, runner TIDAK membangun effective query
    secara manual. Turn kedua dikirim ke generate() bersama history
    agar build_effective_query() dipanggil melalui jalur runtime.
    """

    raw_query = item["query"]

    if isinstance(raw_query, str):

        return {
            "query": raw_query,
            "history": None,
            "is_multi_turn": False,
        }

    if isinstance(raw_query, list):

        if len(raw_query) != 2:
            raise ValueError(
                f"{item['id']}: multi-turn harus tepat "
                f"memiliki 2 turn."
            )

        turn_1 = raw_query[0]
        turn_2 = raw_query[1]

        if not isinstance(turn_1, str):
            raise TypeError(
                f"{item['id']}: turn_1 harus string."
            )

        if not isinstance(turn_2, str):
            raise TypeError(
                f"{item['id']}: turn_2 harus string."
            )

        turn_1 = turn_1.strip()
        turn_2 = turn_2.strip()

        if not turn_1 or not turn_2:
            raise ValueError(
                f"{item['id']}: turn multi-turn tidak boleh kosong."
            )

        history = [
            {
                "role": "user",
                "content": turn_1,
            }
        ]

        return {
            "query": turn_2,
            "history": history,
            "is_multi_turn": True,
        }

    raise TypeError(
        f"{item['id']}: field 'query' harus berupa "
        f"string atau list 2 turn."
    )


# ==============================================================
# RETRIEVAL IDS
# ==============================================================

def extract_retrieved_mal_ids(result):
    """
    Mengambil mal_id dari result['results'].

    Struktur setiap hasil retrieval:
        {
            "rank": ...,
            "mal_id": ...,
            "similarity": ...,
            "text": ...,
            "metadata": {...}
        }
    """

    results = result.get("results", [])

    if not results:
        return []

    mal_ids = []

    for item in results:
        mal_id = item.get("mal_id")

        if mal_id is not None:
            mal_ids.append(mal_id)

    return mal_ids


# ==============================================================
# RESULT RECORD
# ==============================================================

def build_record(
    item,
    condition_name,
    prepared,
    result,
    latency_seconds,
):
    """
    Membuat satu record hasil eksperimen.
    """

    return {
        "query_id": item["id"],
        "category": item["category"],
        "condition": condition_name,

        # Query yang benar-benar diberikan ke generate().
        "query": prepared["query"],

        # Effective query yang benar-benar digunakan pipeline.
        "effective_query": result.get(
            "effective_query"
        ),

        "latency_seconds": latency_seconds,

        # Output generation.
        "answer": result.get(
            "response",
            ""
        ),

        # Retrieval information.
        "retrieved_mal_ids": (
            extract_retrieved_mal_ids(result)
            if condition_name == "B"
            else []
        ),

        # Guardrail.
        "blocked": result.get(
            "blocked",
            False,
        ),

        "refusal": result.get(
            "refusal"
        ),

        # Diagnostic fields.
        "retrieval": result.get(
            "retrieval"
        ),

        "context": result.get(
            "context",
            ""
        ),

        "is_multi_turn": prepared[
            "is_multi_turn"
        ],
    }


# ==============================================================
# MAIN EXPERIMENT
# ==============================================================

def main():

    print("=" * 70)
    print("AniRAG-v2 — PHASE 10 ABLATION STUDY")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. LOAD PIPELINE
    # ----------------------------------------------------------

    print("\n[1] Memuat RagPipeline...")

    pipe = RagPipeline()

    pipe.load_llm(
        quantize=True
    )

    print("[OK] Pipeline dan SLM siap.")

    # ----------------------------------------------------------
    # 2. LOAD TEST SET
    # ----------------------------------------------------------

    queries = load_test_set()

    total_queries = len(queries)
    total_conditions = len(CONDITIONS)
    total_calls = (
        total_queries
        * total_conditions
    )

    print(
        f"\n[INFO] Test set : {total_queries} query"
    )

    print(
        f"[INFO] Kondisi  : {total_conditions} (A/B)"
    )

    print(
        f"[INFO] Total generate() calls: {total_calls}"
    )

    print(
        f"[INFO] max_new_tokens = {MAX_NEW_TOKENS}"
    )

    print(
        f"[INFO] temperature    = {TEMPERATURE}"
    )

    print(
        "[INFO] Condition C tidak dijalankan."
    )

    # ----------------------------------------------------------
    # 3. OUTPUT DIRECTORY
    # ----------------------------------------------------------

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 4. RUN
    # ----------------------------------------------------------

    n_written = 0

    with open(
        OUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for index, item in enumerate(
            queries,
            start=1,
        ):

            prepared = prepare_query(item)

            category = item["category"]

            print(
                f"\n[{index}/{total_queries}] "
                f"{item['id']} ({category})"
            )

            if prepared["is_multi_turn"]:

                print(
                    "  [MULTI-TURN]"
                )

                print(
                    f"  turn_1: "
                    f"{prepared['history'][0]['content']}"
                )

                print(
                    f"  turn_2: "
                    f"{prepared['query']}"
                )

            # --------------------------------------------------
            # A/B
            # --------------------------------------------------

            for condition_name, condition_kwargs in CONDITIONS:

                print(
                    f"  -> Condition {condition_name}"
                )

                try:
                    started = time.perf_counter()

                    result = pipe.generate(
                        prepared["query"],
                        history=prepared["history"],
                        top_k=pipe.retrieval_top_k,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        **condition_kwargs,
                    )

                    latency_seconds = (
                        time.perf_counter() - started
                    )

                except Exception as exc:
                    print(
                        f"     [ERROR] "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                record = build_record(
                    item=item,
                    condition_name=condition_name,
                    prepared=prepared,
                    result=result,
                    latency_seconds=latency_seconds,
                )

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                f.flush()

                n_written += 1

                if result.get("blocked"):

                    print(
                        "     [BLOCKED] "
                        f"{result.get('refusal')}"
                    )

                else:

                    print(
                        "     [OK] "
                        f"effective_query="
                        f"{result.get('effective_query')}"
                    )

                    if condition_name == "B":

                        retrieved_ids = (
                            record[
                                "retrieved_mal_ids"
                            ]
                        )

                        print(
                            "     retrieved="
                            f"{retrieved_ids[:5]}"
                            + (
                                " ..."
                                if len(retrieved_ids) > 5
                                else ""
                            )
                        )

    # ----------------------------------------------------------
    # 5. SUMMARY
    # ----------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "[OK] Eksperimen selesai."
    )

    print(
        f"[OK] {n_written} hasil tersimpan di:"
    )

    print(
        f"     {OUT_PATH}"
    )

    print(
        f"[INFO] Expected records: {total_calls}"
    )

    print(
        f"[INFO] Written records : {n_written}"
    )

    if n_written != total_calls:

        print(
            "[WARNING] Jumlah record tidak sesuai "
            "dengan jumlah generate() yang diharapkan."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()