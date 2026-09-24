"""
AniRAG-v2
PHASE 12 — LLM-as-a-Judge v2.2

Tujuan:
    Mengevaluasi kualitas jawaban generator Llama-3.2-3B-Instruct
    menggunakan model judge yang berbeda, yaitu Gemini.

Eksperimen:
    Kondisi A = SLM-only
    Kondisi B = SLM + RAG

Evaluasi utama:
    Hanya 200 kasus retrieval-dependent:
        - similarity
        - factual
        - attribute_filtering
        - multi_turn_refinement

Karena setiap kasus memiliki kondisi A dan B:
    200 kasus × 2 kondisi = 400 response.

100 kasus:
    - out_of_scope
    - adversarial

tidak digunakan dalam evaluasi kualitas A/B utama.

Rubrik:
    1. relevance
    2. factual_accuracy
    3. coherence

Prinsip v2.2:
    - Relevance menilai intent dan constraint user.
    - Factual accuracy menilai factual claims yang dibuat jawaban.
    - Setiap factual claim dikategorikan:
        SUPPORTED
        CONTRADICTED
        UNSUPPORTED
    - Tidak adanya kontradiksi saja TIDAK cukup untuk factual_accuracy=5.
    - Factual claim utama harus didukung oleh KB untuk memperoleh
      factual_accuracy=5.
    - Constraint/filter failure adalah masalah relevance, bukan
      otomatis factual accuracy.
    - Similarity terutama dinilai sebagai relevance.
    - Klaim metadata konkret tetap harus diverifikasi terhadap KB.
    - Judge hanya boleh menggunakan evidence KB yang diberikan.
    - Judge tidak boleh menggunakan pengetahuan eksternal.
    - "Saya tidak yakin" tanpa factual claim salah tidak otomatis
      menurunkan factual_accuracy.

Calibration:
    --mode calibrate
    default 12 samples, balanced A/B.

Full:
    --mode full

Output:
    results/judge_calibration_v22.jsonl
    results/judge_scores_v22.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_SET_PATH = PROJECT_ROOT / "tests" / "test_set_v1.jsonl"
EXPERIMENT_PATH = PROJECT_ROOT / "results" / "experiment_v1.jsonl"

OUT_PATH = PROJECT_ROOT / "results" / "judge_scores_v22.jsonl"
CALIBRATION_OUT_PATH = (
    PROJECT_ROOT / "results" / "judge_calibration_v22.jsonl"
)


# ============================================================
# JUDGE CONFIGURATION
# ============================================================

DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 2
REQUEST_DELAY_SECONDS = 6

EXPECTED_EXPERIMENT_RECORDS = 600
EXPECTED_RETRIEVAL_RECORDS = 400
EXPECTED_KB_ROWS = 15966
EXPECTED_CALIBRATION_CASES = 12


# ============================================================
# FINAL KB SCHEMA
# ============================================================

KB_FIELDS = [
    "mal_id",
    "title",
    "title_english",
    "type",
    "episodes",
    "score",
    "genres",
    "synopsis",
    "year",
    "themes",
    "studios",
    "image_url",
]

REFERENCE_FIELDS = [
    "mal_id",
    "title",
    "title_english",
    "type",
    "episodes",
    "score",
    "genres",
    "synopsis",
    "year",
    "themes",
    "studios",
]


RETRIEVAL_CATEGORIES = {
    "similarity",
    "factual",
    "attribute_filtering",
    "multi_turn_refinement",
}


# ============================================================
# GEMINI SAFETY ERROR
# ============================================================

class GeminiProhibitedContentError(RuntimeError):
    """
    Gemini memblokir prompt karena safety filter
    PROHIBITED_CONTENT.
    """

    pass


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalisasi teks untuk pencocokan title/query.
    """
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# JSONL HELPERS
# ============================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Membaca JSONL dan mengembalikan list dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada {path}, baris {line_number}: {exc}"
                ) from exc

    return records


def write_jsonl(
    path: Path,
    records: List[Dict[str, Any]],
) -> None:
    """
    Menulis list dictionary ke JSONL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# DATASET LOADER
# ============================================================

def load_kb_records() -> List[Dict[str, Any]]:
    """
    Membaca final cleaned anime dataset.
    """

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "dataset_anime_clean.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset_path}"
        )

    import pandas as pd

    df = pd.read_csv(dataset_path)

    records = df.to_dict(orient="records")

    if len(records) != EXPECTED_KB_ROWS:
        print(
            "[WARN] Jumlah KB berbeda dari expected:"
            f" {len(records)} != {EXPECTED_KB_ROWS}"
        )

    return records


# ============================================================
# TEST SET
# ============================================================

def load_test_set() -> List[Dict[str, Any]]:
    """
    Membaca test_set_v1.jsonl.
    """

    records = read_jsonl(TEST_SET_PATH)

    required_fields = {
        "id",
        "category",
        "query",
    }

    for index, record in enumerate(records, start=1):
        missing = required_fields - set(record.keys())

        if missing:
            raise ValueError(
                f"Test case #{index} kekurangan field: {sorted(missing)}"
            )

    return records


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

def load_experiment_results() -> List[Dict[str, Any]]:
    """
    Membaca hasil Phase 10.
    """

    records = read_jsonl(EXPERIMENT_PATH)

    if len(records) != EXPECTED_EXPERIMENT_RECORDS:
        raise ValueError(
            "Jumlah experiment records tidak sesuai: "
            f"{len(records)} != {EXPECTED_EXPERIMENT_RECORDS}"
        )

    return records


# ============================================================
# TITLE INDEX
# ============================================================

def build_title_index(
    kb_records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Index title dan title_english untuk pencocokan entity.
    """

    title_index: Dict[str, List[Dict[str, Any]]] = {}

    for record in kb_records:
        mal_id = record.get("mal_id")

        for field in ("title", "title_english"):
            value = record.get(field)

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            normalized = normalize_text(value)

            title_index.setdefault(
                normalized,
                [],
            ).append(
                {
                    "mal_id": mal_id,
                    "title": record.get("title"),
                    "title_english": record.get("title_english"),
                    "field": field,
                }
            )

    return title_index


# ============================================================
# GROUND TRUTH
# ============================================================

def get_ground_truth_ids(
    test_case: Dict[str, Any],
) -> List[Any]:
    """
    Mengambil ground truth MAL IDs jika tersedia.
    """

    ground_truth = test_case.get("ground_truth_mal_ids")

    if not ground_truth:
        return []

    return list(ground_truth)


# ============================================================
# REFERENCE RECORD LOOKUP
# ============================================================

def build_kb_by_mal_id(
    kb_records: List[Dict[str, Any]],
) -> Dict[Any, Dict[str, Any]]:
    """
    Index KB berdasarkan mal_id.
    """

    kb_by_id: Dict[Any, Dict[str, Any]] = {}

    for record in kb_records:
        mal_id = record.get("mal_id")

        if mal_id is None:
            continue

        kb_by_id[mal_id] = record

    return kb_by_id


# ============================================================
# TITLE MATCHING
# ============================================================

def find_title_matches(
    query: str,
    kb_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Mencari anime yang title/title_english-nya muncul
    secara eksplisit di query.
    """

    normalized_query = normalize_text(query)

    matches: List[Dict[str, Any]] = []

    for record in kb_records:
        for field in ("title", "title_english"):
            value = record.get(field)

            if value is None:
                continue

            title = str(value).strip()

            if not title:
                continue

            normalized_title = normalize_text(title)

            if not normalized_title:
                continue

            pattern = (
                r"(?<!\w)"
                + re.escape(normalized_title)
                + r"(?!\w)"
            )

            if re.search(pattern, normalized_query):
                matches.append(
                    {
                        "mal_id": record.get("mal_id"),
                        "title": record.get("title"),
                        "title_english": record.get("title_english"),
                        "matched_field": field,
                    }
                )

    return matches


# ============================================================
# REFERENCE EVIDENCE
# ============================================================

def build_reference_evidence(
    experiment_record: Dict[str, Any],
    test_case: Dict[str, Any],
    kb_records: List[Dict[str, Any]],
    title_index: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Membentuk evidence KB yang digunakan sebagai reference
    untuk LLM-as-a-Judge.

    Factual:
      ground_truth_mal_ids menjadi sumber utama.

    Non-factual:
      menggunakan title match dan ground truth jika tersedia.
    """

    category = str(
        experiment_record.get(
            "category",
            test_case.get("category", ""),
        )
    )

    query = str(
        experiment_record.get(
            "query",
            test_case.get("query", ""),
        )
    )

    kb_by_id = build_kb_by_mal_id(kb_records)

    ground_truth_ids = get_ground_truth_ids(test_case)

    selected_ids: List[Any] = []

    # --------------------------------------------------------
    # Factual
    # --------------------------------------------------------

    if category == "factual":
        if ground_truth_ids:
            selected_ids.extend(ground_truth_ids[:3])

        else:
            title_matches = find_title_matches(
                query=query,
                kb_records=kb_records,
            )

            for match in title_matches:
                mal_id = match.get("mal_id")

                if mal_id is not None:
                    selected_ids.append(mal_id)

                if len(selected_ids) >= 3:
                    break

    # --------------------------------------------------------
    # Non-factual
    # --------------------------------------------------------

    else:
        title_matches = find_title_matches(
            query=query,
            kb_records=kb_records,
        )

        for match in title_matches:
            mal_id = match.get("mal_id")

            if mal_id is not None:
                selected_ids.append(mal_id)

        for mal_id in ground_truth_ids:
            selected_ids.append(mal_id)

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    unique_ids: List[Any] = []

    for mal_id in selected_ids:
        if mal_id not in unique_ids:
            unique_ids.append(mal_id)

    unique_ids = unique_ids[:12]

    records: List[Dict[str, Any]] = []

    for mal_id in unique_ids:
        record = kb_by_id.get(mal_id)

        if record is None:
            continue

        reference_record: Dict[str, Any] = {}

        for field in REFERENCE_FIELDS:
            if field in record:
                reference_record[field] = record[field]

        records.append(reference_record)

    return {
        "category": category,
        "records": records,
        "ground_truth_mal_ids": ground_truth_ids,
        "ground_truth_reference_used": bool(ground_truth_ids),
        "evidence_mode": "standard",
    }


# ============================================================
# FACTUAL FIELD DETECTION
# ============================================================

def detect_factual_fields(
    query: str,
) -> List[str]:
    """
    Mendeteksi field KB yang secara eksplisit diminta
    oleh query.

    Digunakan hanya untuk safety fallback apabila
    prompt standard diblokir Gemini.
    """

    text = normalize_text(query)

    field_patterns = [
        (
            "score",
            [
                r"\bscore\b",
                r"\bskor\b",
                r"\bnilai\b",
                r"\brating\b",
            ],
        ),
        (
            "year",
            [
                r"\btahun\b",
                r"\bkeluar tahun\b",
                r"\brilis tahun\b",
            ],
        ),
        (
            "episodes",
            [
                r"\bepisode\b",
                r"\bepisodenya\b",
                r"\bjumlah episode\b",
            ],
        ),
        (
            "type",
            [
                r"\btipe\b",
                r"\btype\b",
                r"\bformat\b",
            ],
        ),
        (
            "genres",
            [
                r"\bgenre\b",
                r"\bgenrenya\b",
            ],
        ),
        (
            "themes",
            [
                r"\btema\b",
                r"\btheme\b",
                r"\btemanya\b",
            ],
        ),
        (
            "studios",
            [
                r"\bstudio\b",
                r"\bstudionya\b",
            ],
        ),
        (
            "title_english",
            [
                r"\bjudul inggris\b",
                r"\bjudul english\b",
                r"\benglish title\b",
            ],
        ),
        (
            "title",
            [
                r"\bjudul\b",
                r"\btitle\b",
                r"\bnama anime\b",
            ],
        ),
    ]

    detected: List[str] = []

    for field, patterns in field_patterns:
        if any(
            re.search(pattern, text)
            for pattern in patterns
        ):
            detected.append(field)

    return detected


# ============================================================
# SAFE REFERENCE EVIDENCE
# ============================================================

def build_safe_reference_evidence(
    reference_evidence: Dict[str, Any],
    query: str,
) -> Dict[str, Any]:
    """
    Membatasi reference evidence ketika Gemini memblokir
    prompt standard karena PROHIBITED_CONTENT.

    Synopsis sengaja tidak disertakan.
    Hanya title, mal_id, dan field faktual yang relevan
    dengan query yang dipertahankan.
    """

    factual_fields = detect_factual_fields(query)

    allowed_fields = {
        "mal_id",
        "title",
    }

    allowed_fields.update(factual_fields)

    safe_records: List[Dict[str, Any]] = []

    for record in reference_evidence.get(
        "records",
        [],
    ):
        safe_record: Dict[str, Any] = {}

        for field in REFERENCE_FIELDS:
            if field not in allowed_fields:
                continue

            if field not in record:
                continue

            safe_record[field] = record[field]

        safe_records.append(safe_record)

    return {
        "category": reference_evidence.get(
            "category"
        ),
        "records": safe_records,
        "ground_truth_mal_ids": reference_evidence.get(
            "ground_truth_mal_ids",
            [],
        ),
        "ground_truth_reference_used": reference_evidence.get(
            "ground_truth_reference_used",
            False,
        ),
        "evidence_mode": "safe_factual_fallback",
    }


# ============================================================
# ACTUAL RETRIEVAL EVIDENCE
# ============================================================

def extract_actual_retrieval_evidence(
    experiment_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Mengambil evidence retrieval aktual dari hasil eksperimen.
    """

    retrieval = experiment_record.get(
        "retrieval"
    )

    return {
        "retrieval": retrieval,
        "retrieved_mal_ids": experiment_record.get(
            "retrieved_mal_ids",
            [],
        ),
        "context": experiment_record.get(
            "context",
            "",
        ),
    }


# ============================================================
# SAFE ACTUAL RETRIEVAL EVIDENCE
# ============================================================

def build_safe_actual_retrieval_evidence(
    experiment_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Membentuk evidence retrieval versi aman.

    Synopsis tidak disertakan.
    Field metadata yang dipertahankan dibatasi pada:
      - mal_id
      - title
      - field yang relevan dengan query
    """

    query = str(
        experiment_record.get(
            "query",
            "",
        )
    )

    factual_fields = detect_factual_fields(
        query
    )

    allowed_metadata_fields = {
        "mal_id",
        "title",
    }

    allowed_metadata_fields.update(
        factual_fields
    )

    retrieval = experiment_record.get(
        "retrieval"
    ) or {}

    safe_retrieval: Dict[str, Any] = {}

    if isinstance(retrieval, dict):

        for key in (
            "query",
            "filters",
            "anchor",
            "factual_entity",
            "retrieval_mode",
            "candidate_k",
            "hard_filters_active",
            "candidate_count",
            "after_attribute_filter",
            "after_anchor_exclusion",
        ):
            if key in retrieval:
                safe_retrieval[key] = retrieval[key]

        safe_results: List[Dict[str, Any]] = []

        for result in retrieval.get(
            "results",
            []
        ) or []:

            if not isinstance(
                result,
                dict,
            ):
                continue

            safe_result: Dict[str, Any] = {}

            for key in (
                "rank",
                "mal_id",
                "similarity",
                "final_rank",
                "exact_match",
                "retrieval_score",
            ):
                if key in result:
                    safe_result[key] = result[key]

            metadata = result.get(
                "metadata"
            )

            if isinstance(
                metadata,
                dict,
            ):
                safe_metadata: Dict[str, Any] = {}

                for field in allowed_metadata_fields:
                    if field in metadata:
                        safe_metadata[field] = metadata[field]

                if safe_metadata:
                    safe_result[
                        "metadata"
                    ] = safe_metadata

            safe_results.append(
                safe_result
            )

        safe_retrieval[
            "results"
        ] = safe_results

    return {
        "retrieval": safe_retrieval,
        "retrieved_mal_ids": experiment_record.get(
            "retrieved_mal_ids",
            [],
        ),
        "context": (
            "[Context disederhanakan untuk "
            "safety fallback; synopsis tidak "
            "disertakan.]"
        ),
        "evidence_mode": "safe_factual_fallback",
    }


# ============================================================
# JUDGE PROMPT
# ============================================================

JUDGE_PROMPT = r"""
Anda adalah evaluator untuk eksperimen chatbot rekomendasi anime
berbasis Retrieval-Augmented Generation (RAG) dan Small Language Model
(SLM).

Tugas Anda adalah mengevaluasi SATU jawaban chatbot berdasarkan:
1. query pengguna,
2. kategori test,
3. kondisi eksperimen,
4. effective query,
5. actual retrieval evidence,
6. reference evidence dari knowledge base.

JANGAN menggunakan pengetahuan eksternal di luar evidence yang diberikan.

============================================================
QUERY
============================================================

{query}

============================================================
KATEGORI
============================================================

{category}

============================================================
KONDISI
============================================================

{condition}

============================================================
EFFECTIVE QUERY
============================================================

{effective_query}

============================================================
JAWABAN CHATBOT
============================================================

{answer}

============================================================
ACTUAL RETRIEVAL EVIDENCE
============================================================

{actual_retrieval_evidence}

============================================================
REFERENCE EVIDENCE
============================================================

{reference_evidence}

============================================================
TUJUAN PENILAIAN
============================================================

Nilai tiga dimensi:

1. RELEVANCE
2. FACTUAL ACCURACY
3. COHERENCE

Masing-masing diberi nilai integer 1 sampai 5.

============================================================
RELEVANCE
============================================================

Nilai apakah jawaban menjawab kebutuhan pengguna dan mengikuti
constraint yang diminta.

5:
- Sangat relevan.
- Menjawab permintaan secara langsung.
- Constraint terpenuhi.

4:
- Relevan dengan kekurangan kecil.

3:
- Sebagian relevan tetapi ada informasi penting yang kurang.

2:
- Sebagian besar tidak memenuhi kebutuhan pengguna.

1:
- Tidak menjawab pertanyaan atau keluar dari permintaan.

Untuk query rekomendasi dengan constraint seperti genre, tema,
score minimum, tahun, studio, atau atribut lain, kegagalan memenuhi
constraint terutama menurunkan RELEVANCE.

Untuk similarity:
- fokus utama adalah apakah rekomendasi sesuai dengan maksud
  similarity pengguna.
- Jangan menilai similarity hanya berdasarkan title existence.
- Metadata konkret yang disebutkan dalam jawaban tetap harus
  diperiksa untuk FACTUAL ACCURACY.

============================================================
FACTUAL ACCURACY
============================================================

Nilai apakah klaim faktual dapat didukung oleh evidence.

Setiap klaim faktual dapat dianggap:

SUPPORTED
CONTRADICTED
UNSUPPORTED

SUPPORTED:
- Secara eksplisit didukung oleh reference evidence atau
  actual retrieval evidence.

CONTRADICTED:
- Bertentangan dengan evidence.

UNSUPPORTED:
- Tidak terdapat dukungan dari evidence yang tersedia.

PENTING:

- Jangan menggunakan pengetahuan eksternal.
- Jangan menganggap sebuah title otomatis memvalidasi synopsis.
- Jangan menganggap sebuah anime benar hanya karena title-nya
  terdapat dalam KB.
- Metadata seperti score, year, episodes, type, genres, themes,
  dan studios harus dibandingkan dengan evidence.
- Beberapa klaim unsupported harus menurunkan factual accuracy.
- Factual accuracy = 5 hanya jika klaim faktual yang dibuat
  memiliki dukungan evidence yang memadai.
- Jawaban "saya tidak tahu" atau "saya tidak yakin" tidak otomatis
  memiliki factual accuracy rendah apabila jawaban tersebut tidak
  membuat klaim faktual yang salah.

============================================================
COHERENCE
============================================================

Nilai kualitas penyampaian jawaban.

5:
- Jelas.
- Konsisten.
- Ringkas.
- Mudah dipahami.
- Tidak kontradiktif.

4:
- Jelas dengan sedikit kekurangan.

3:
- Masih dapat dipahami tetapi kurang rapi.

2:
- Sulit dipahami atau tidak konsisten.

1:
- Sangat tidak jelas atau tidak dapat dipahami.

============================================================
KHUSUS RETRIEVAL
============================================================

Gunakan actual retrieval evidence untuk memahami apa yang benar-benar
ditemukan oleh sistem.

Gunakan reference evidence sebagai pembanding terhadap ground truth.

Jika query faktual meminta atribut tertentu, misalnya score:
- fokus factual accuracy pada nilai score yang dijawab.
- Jangan menghukum jawaban karena tidak memberikan synopsis jika
  pengguna hanya meminta score.

Jika query meminta beberapa constraint:
- constraint failure terutama memengaruhi relevance.
- Jika chatbot mengklaim bahwa semua constraint terpenuhi padahal
  evidence menunjukkan sebaliknya, factual accuracy juga dapat turun.

============================================================
OUTPUT
============================================================

Kembalikan HANYA JSON valid berikut:

{
  "relevance": 1,
  "factual_accuracy": 1,
  "coherence": 1,
  "reasoning": "Alasan singkat dan berbasis evidence."
}

Semua nilai harus integer 1-5.

Jangan menggunakan markdown.
Jangan menambahkan field lain.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    experiment_record: Dict[str, Any],
    reference_evidence: Dict[str, Any],
    actual_retrieval_evidence: Dict[str, Any],
) -> str:
    prompt = JUDGE_PROMPT

    replacements = {
        "{query}": str(
            experiment_record.get("query", "")
        ),
        "{category}": str(
            experiment_record.get("category", "")
        ),
        "{condition}": str(
            experiment_record.get("condition", "")
        ),
        "{effective_query}": str(
            experiment_record.get("effective_query", "")
        ),
        "{answer}": str(
            experiment_record.get("answer", "")
        ),
        "{actual_retrieval_evidence}": json.dumps(
            actual_retrieval_evidence,
            ensure_ascii=False,
            indent=2,
        ),
        "{reference_evidence}": json.dumps(
            reference_evidence,
            ensure_ascii=False,
            indent=2,
        ),
    }

    for placeholder, value in replacements.items():
        prompt = prompt.replace(
            placeholder,
            value,
        )

    return prompt


# ============================================================
# GEMINI RETRY HELPERS
# ============================================================

def extract_retry_delay_seconds(
    error: Exception,
) -> Optional[float]:
    """
    Mengambil retryDelay dari error Gemini jika tersedia.
    """

    error_text = str(error)

    patterns = [
        r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s",
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",
        r"retry in\s+(\d+(?:\.\d+)?)s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            error_text,
            flags=re.IGNORECASE,
        )

        if match:

            try:
                return float(
                    match.group(1)
                )

            except ValueError:
                pass

    return None


def is_retryable_gemini_error(
    error: Exception,
) -> bool:
    """
    Menentukan apakah error Gemini layak di-retry.
    """

    error_text = str(error).lower()

    retryable_patterns = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "response tanpa text",
        "without text",
        "503",
        "service unavailable",
        "internal server error",
        "temporarily unavailable",
    ]

    return any(
        pattern in error_text
        for pattern in retryable_patterns
    )


# ============================================================
# GEMINI PROHIBITED CONTENT DETECTOR
# ============================================================

def is_prohibited_content_response(
    response: Any,
) -> bool:
    """
    Mendeteksi apakah Gemini memblokir request karena
    PROHIBITED_CONTENT.
    """

    prompt_feedback = getattr(
        response,
        "prompt_feedback",
        None,
    )

    if prompt_feedback is None:
        return False

    block_reason = getattr(
        prompt_feedback,
        "block_reason",
        None,
    )

    if block_reason is None:
        return False

    return (
        str(block_reason).upper()
        == "PROHIBITED_CONTENT"
        or
        "PROHIBITED_CONTENT"
        in str(block_reason).upper()
    )


# ============================================================
# GEMINI API
# ============================================================

def call_gemini(
    prompt: str,
    model_name: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Memanggil Gemini dengan retry handling.

    PROHIBITED_CONTENT tidak di-retry karena retry prompt yang sama
    tidak akan menyelesaikan safety block.
    """

    from google import genai

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY belum tersedia "
            "di environment variable."
        )

    client = genai.Client(
        api_key=api_key
    )

    last_error: Optional[Exception] = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            # ------------------------------------------------
            # SAFETY BLOCK
            # ------------------------------------------------

            if is_prohibited_content_response(
                response
            ):
                raise GeminiProhibitedContentError(
                    "Gemini memblokir prompt karena "
                    "PROHIBITED_CONTENT."
                )

            text = getattr(
                response,
                "text",
                None,
            )

            if not text or not text.strip():

                raise RuntimeError(
                    "Gemini mengembalikan response "
                    "tanpa text."
                )

            return text.strip()

        except GeminiProhibitedContentError:

            # Jangan retry prompt yang sama.
            raise

        except Exception as exc:

            last_error = exc

            print(
                f"[WARN] Gemini attempt {attempt} gagal: {exc}"
            )

            if attempt >= max_retries:
                break

            if not is_retryable_gemini_error(
                exc
            ):

                print(
                    "[ERROR] Error Gemini tidak dianggap "
                    "retryable. Tidak melakukan retry."
                )

                break

            server_retry_delay = (
                extract_retry_delay_seconds(
                    exc
                )
            )

            if server_retry_delay is not None:

                sleep_seconds = (
                    server_retry_delay
                    + 1.0
                )

                print(
                    "[INFO] Gemini meminta retry "
                    f"setelah {server_retry_delay:.0f}s."
                )

                print(
                    f"[INFO] Menunggu {sleep_seconds:.0f}s "
                    "sebelum retry..."
                )

            else:

                sleep_seconds = (
                    RETRY_BASE_SECONDS
                    * (2 ** (attempt - 1))
                )

                print(
                    "[INFO] retryDelay tidak tersedia. "
                    f"Exponential backoff: "
                    f"{sleep_seconds}s..."
                )

            time.sleep(
                sleep_seconds
            )

    raise RuntimeError(
        f"Gemini gagal setelah "
        f"{max_retries} percobaan: "
        f"{last_error}"
    )


# ============================================================
# JUDGE JSON PARSER
# ============================================================

def parse_judge_json(
    raw_response: str,
) -> Dict[str, Any]:
    """
    Parse dan validasi JSON hasil Gemini.
    """

    text = raw_response.strip()

    # --------------------------------------------------------
    # Remove accidental markdown fence
    # --------------------------------------------------------

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    try:

        data = json.loads(
            text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Response judge bukan JSON valid: "
            f"{raw_response}"
        ) from exc

    required_fields = {
        "relevance",
        "factual_accuracy",
        "coherence",
        "reasoning",
    }

    missing = (
        required_fields
        - set(data.keys())
    )

    if missing:

        raise ValueError(
            "Field judge kurang: "
            f"{sorted(missing)}"
        )

    for field in (
        "relevance",
        "factual_accuracy",
        "coherence",
    ):

        value = data[field]

        if (
            not isinstance(
                value,
                int,
            )
            or isinstance(
                value,
                bool,
            )
        ):

            raise ValueError(
                f"{field} harus integer."
            )

        if not 1 <= value <= 5:

            raise ValueError(
                f"{field} harus berada "
                "pada rentang 1-5."
            )

    if not isinstance(
        data["reasoning"],
        str,
    ):

        raise ValueError(
            "reasoning harus berupa string."
        )

    return {
        "relevance": data["relevance"],
        "factual_accuracy": data[
            "factual_accuracy"
        ],
        "coherence": data["coherence"],
        "reasoning": data["reasoning"].strip(),
    }


# ============================================================
# RETRIEVAL CASE FILTER
# ============================================================

def filter_retrieval_cases(
    experiment_records: List[Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Mengambil hanya retrieval-dependent cases:
      similarity
      factual
      attribute_filtering
      multi_turn_refinement

    Target:
      200 query
      400 A/B records
    """

    test_case_by_id = {
        str(item["id"]): item
        for item in test_cases
    }

    retrieval_records: List[Dict[str, Any]] = []

    for record in experiment_records:

        category = str(
            record.get(
                "category",
                "",
            )
        )

        if category not in RETRIEVAL_CATEGORIES:
            continue

        query_id = str(
            record.get(
                "query_id",
                "",
            )
        )

        if query_id not in test_case_by_id:
            raise ValueError(
                f"Query ID {query_id} "
                "tidak ditemukan di test set."
            )

        retrieval_records.append(
            record
        )

    if len(retrieval_records) != EXPECTED_RETRIEVAL_RECORDS:

        raise ValueError(
            "Jumlah retrieval records tidak sesuai: "
            f"{len(retrieval_records)} != "
            f"{EXPECTED_RETRIEVAL_RECORDS}"
        )

    return retrieval_records


# ============================================================
# A/B COMPLETENESS
# ============================================================

def validate_ab_completeness(
    records: List[Dict[str, Any]],
) -> None:
    """
    Memastikan setiap query retrieval memiliki
    condition A dan B.
    """

    grouped: Dict[str, set] = {}

    for record in records:

        query_id = str(
            record.get(
                "query_id",
                "",
            )
        )

        condition = str(
            record.get(
                "condition",
                "",
            )
        )

        grouped.setdefault(
            query_id,
            set(),
        ).add(condition)

    incomplete = {
        query_id: conditions
        for query_id, conditions
        in grouped.items()
        if conditions != {"A", "B"}
    }

    if incomplete:

        raise ValueError(
            "Ada query retrieval tanpa pasangan "
            f"A/B lengkap: {incomplete}"
        )


# ============================================================
# EXISTING JUDGE RESULTS
# ============================================================

def load_existing_judge_results() -> Dict[
    Tuple[str, str],
    Dict[str, Any],
]:
    """
    Membaca hasil judge yang sudah tersedia.

    Digunakan untuk resume.
    """

    if not OUT_PATH.exists():
        return {}

    records = read_jsonl(
        OUT_PATH
    )

    results: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ] = {}

    for record in records:

        query_id = str(
            record.get(
                "query_id",
                "",
            )
        )

        condition = str(
            record.get(
                "condition",
                "",
            )
        )

        if not query_id or not condition:
            continue

        results[
            (
                query_id,
                condition,
            )
        ] = record

    return results


# ============================================================
# EVALUATE ONE RECORD
# ============================================================

def evaluate_record(
    experiment_record: Dict[str, Any],
    test_case: Dict[str, Any],
    kb_records: List[Dict[str, Any]],
    title_index: Dict[str, List[Dict[str, Any]]],
    judge_model: str,
) -> Dict[str, Any]:
    """
    Mengevaluasi satu record experiment.

    Normal:
      standard evidence

    Jika Gemini memblokir prompt karena
    PROHIBITED_CONTENT:
      safe evidence fallback tanpa synopsis.
    """

    # --------------------------------------------------------
    # STANDARD EVIDENCE
    # --------------------------------------------------------

    reference_evidence = build_reference_evidence(
        experiment_record=experiment_record,
        test_case=test_case,
        kb_records=kb_records,
        title_index=title_index,
    )

    actual_retrieval_evidence = (
        extract_actual_retrieval_evidence(
            experiment_record
        )
    )

    prompt = build_prompt(
        experiment_record=experiment_record,
        reference_evidence=reference_evidence,
        actual_retrieval_evidence=actual_retrieval_evidence,
    )

    # --------------------------------------------------------
    # GEMINI CALL
    # --------------------------------------------------------

    try:

        raw_response = call_gemini(
            prompt=prompt,
            model_name=judge_model,
        )

    except GeminiProhibitedContentError:

        print(
            "[WARN] Gemini memblokir prompt karena "
            "PROHIBITED_CONTENT."
        )

        print(
            "[INFO] Menggunakan safe evidence fallback "
            "tanpa synopsis."
        )

        # ----------------------------------------------------
        # SAFE REFERENCE
        # ----------------------------------------------------

        safe_reference_evidence = (
            build_safe_reference_evidence(
                reference_evidence=reference_evidence,
                query=str(
                    experiment_record.get(
                        "query",
                        "",
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # SAFE ACTUAL RETRIEVAL
        # ----------------------------------------------------

        safe_actual_retrieval = (
            build_safe_actual_retrieval_evidence(
                experiment_record=experiment_record,
            )
        )

        # ----------------------------------------------------
        # SAFE PROMPT
        # ----------------------------------------------------

        safe_prompt = JUDGE_PROMPT

        safe_replacements = {
            "{query}": str(
                experiment_record.get(
                    "query",
                    "",
                )
            ),
            "{category}": str(
                experiment_record.get(
                    "category",
                    "",
                )
            ),
            "{condition}": str(
                experiment_record.get(
                    "condition",
                    "",
                )
            ),
            "{effective_query}": str(
                experiment_record.get(
                    "effective_query",
                    "",
                )
            ),
            "{answer}": str(
                experiment_record.get(
                    "answer",
                    "",
                )
            ),
            "{actual_retrieval_evidence}": json.dumps(
                safe_actual_retrieval,
                ensure_ascii=False,
                indent=2,
            ),
            "{reference_evidence}": json.dumps(
                safe_reference_evidence,
                ensure_ascii=False,
                indent=2,
            ),
        }

        for placeholder, value in safe_replacements.items():
            safe_prompt = safe_prompt.replace(
                placeholder,
                value,
            )

        safe_prompt += (
            "\n\n"
            "CATATAN EVALUATOR: Evidence di atas adalah "
            "safe factual evidence yang sengaja dibatasi "
            "hanya pada field relevan dengan pertanyaan. "
            "Jangan menganggap field yang tidak ditampilkan "
            "sebagai bukti tersedia."
        )

        # ----------------------------------------------------
        # SECOND GEMINI CALL
        # ----------------------------------------------------

        raw_response = call_gemini(
            prompt=safe_prompt,
            model_name=judge_model,
        )

    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    judge_result = parse_judge_json(
        raw_response
    )

    # --------------------------------------------------------
    # FINAL RECORD
    # --------------------------------------------------------

    return {
        "query_id": experiment_record.get(
            "query_id"
        ),
        "category": experiment_record.get(
            "category"
        ),
        "condition": experiment_record.get(
            "condition"
        ),
        "query": experiment_record.get(
            "query"
        ),
        "answer": experiment_record.get(
            "answer"
        ),
        "relevance": judge_result[
            "relevance"
        ],
        "factual_accuracy": judge_result[
            "factual_accuracy"
        ],
        "coherence": judge_result[
            "coherence"
        ],
        "reasoning": judge_result[
            "reasoning"
        ],
        "judge_model": judge_model,
    }


# ============================================================
# CALIBRATION CASES
# ============================================================

CALIBRATION_QUERY_IDS = [
    "ATTR-001",
    "FACT-001",
    "MULTI-001",
    "SIM-001",
    "SIM-002",
    "FACT-002",
]


def select_calibration_records(
    experiment_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Mengambil 6 query × 2 condition = 12 records.
    """

    selected: List[Dict[str, Any]] = []

    for query_id in CALIBRATION_QUERY_IDS:

        for condition in (
            "A",
            "B",
        ):

            matches = [
                record
                for record in experiment_records
                if str(
                    record.get(
                        "query_id",
                        "",
                    )
                )
                == query_id
                and str(
                    record.get(
                        "condition",
                        "",
                    )
                )
                == condition
            ]

            if not matches:

                raise ValueError(
                    f"Calibration record tidak ditemukan: "
                    f"{query_id}/{condition}"
                )

            selected.append(
                matches[0]
            )

    if len(selected) != EXPECTED_CALIBRATION_CASES:

        raise ValueError(
            "Jumlah calibration case tidak sesuai: "
            f"{len(selected)} != "
            f"{EXPECTED_CALIBRATION_CASES}"
        )

    return selected


# ============================================================
# CALIBRATION RUN
# ============================================================

def run_calibration(
    judge_model: str,
) -> None:
    """
    Menjalankan calibration 12 records.
    """

    print(
        "[INFO] Memulai calibration..."
    )

    test_cases = load_test_set()
    experiment_records = load_experiment_results()
    kb_records = load_kb_records()

    title_index = build_title_index(
        kb_records
    )

    calibration_records = (
        select_calibration_records(
            experiment_records
        )
    )

    test_case_by_id = {
        str(item["id"]): item
        for item in test_cases
    }

    output_records: List[
        Dict[str, Any]
    ] = []

    for index, experiment_record in enumerate(
        calibration_records,
        start=1,
    ):

        query_id = str(
            experiment_record.get(
                "query_id",
                "",
            )
        )

        print(
            f"[CALIBRATION {index}/{len(calibration_records)}] "
            f"{query_id}/"
            f"{experiment_record.get('condition')}"
        )

        test_case = test_case_by_id[
            query_id
        ]

        try:

            result = evaluate_record(
                experiment_record=experiment_record,
                test_case=test_case,
                kb_records=kb_records,
                title_index=title_index,
                judge_model=judge_model,
            )

            output_records.append(
                result
            )

        except Exception as exc:

            print(
                f"[ERROR] Calibration gagal "
                f"{query_id}: {exc}"
            )

        if index < len(
            calibration_records
        ):

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    write_jsonl(
        CALIBRATION_OUT_PATH,
        output_records,
    )

    print(
        "[INFO] Calibration selesai:"
        f" {len(output_records)}/"
        f"{len(calibration_records)}"
    )

    print(
        f"[INFO] Output: {CALIBRATION_OUT_PATH}"
    )


# ============================================================
# FULL EVALUATION
# ============================================================

def run_full(
    judge_model: str,
) -> None:
    """
    Menjalankan full LLM-as-a-Judge retrieval evaluation.

    Resume:
      record yang sudah ada tidak dipanggil ulang.
    """

    print(
        "[INFO] Memulai full LLM-as-a-Judge..."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    test_cases = load_test_set()

    experiment_records = (
        load_experiment_results()
    )

    kb_records = load_kb_records()

    title_index = build_title_index(
        kb_records
    )

    # --------------------------------------------------------
    # FILTER RETRIEVAL CASES
    # --------------------------------------------------------

    retrieval_records = (
        filter_retrieval_cases(
            experiment_records=experiment_records,
            test_cases=test_cases,
        )
    )

    validate_ab_completeness(
        retrieval_records
    )

    print(
        "[INFO] Retrieval records:"
        f" {len(retrieval_records)}"
    )

    # --------------------------------------------------------
    # EXISTING RESULTS
    # --------------------------------------------------------

    results_by_key = (
        load_existing_judge_results()
    )

    print(
        "[INFO] Existing judge results:"
        f" {len(results_by_key)}"
    )

    # --------------------------------------------------------
    # TEST CASE INDEX
    # --------------------------------------------------------

    test_case_by_id = {
        str(item["id"]): item
        for item in test_cases
    }

    # --------------------------------------------------------
    # PENDING
    # --------------------------------------------------------

    pending: List[
        Dict[str, Any]
    ] = []

    for record in retrieval_records:

        query_id = str(
            record.get(
                "query_id",
                "",
            )
        )

        condition = str(
            record.get(
                "condition",
                "",
            )
        )

        key = (
            query_id,
            condition,
        )

        if key not in results_by_key:
            pending.append(
                record
            )

    print(
        "[INFO] Pending judge records:"
        f" {len(pending)}"
    )

    # --------------------------------------------------------
    # NOTHING TO DO
    # --------------------------------------------------------

    if not pending:

        print(
            "[INFO] Semua retrieval records "
            "sudah dinilai."
        )

        ordered_records = sorted(
            results_by_key.values(),
            key=lambda item: (
                str(
                    item.get(
                        "query_id",
                        "",
                    )
                ),
                str(
                    item.get(
                        "condition",
                        "",
                    )
                ),
            ),
        )

        write_jsonl(
            OUT_PATH,
            ordered_records,
        )

        print(
            f"[INFO] Output: {OUT_PATH}"
        )

        return

    # --------------------------------------------------------
    # EVALUATION LOOP
    # --------------------------------------------------------

    for index, record in enumerate(
        pending,
        start=1,
    ):

        query_id = str(
            record.get(
                "query_id",
                "",
            )
        )

        condition = str(
            record.get(
                "condition",
                "",
            )
        )

        print(
            f"[{index}/{len(pending)}] "
            f"{query_id} - {condition}"
        )

        test_case = test_case_by_id.get(
            query_id
        )

        if test_case is None:

            print(
                f"[ERROR] Test case {query_id} "
                "tidak ditemukan."
            )

            continue

        try:

            result = evaluate_record(
                experiment_record=record,
                test_case=test_case,
                kb_records=kb_records,
                title_index=title_index,
                judge_model=judge_model,
            )

            key = (
                query_id,
                condition,
            )

            results_by_key[key] = result

            # ------------------------------------------------
            # SAVE AFTER EACH SUCCESS
            # ------------------------------------------------

            ordered_records = sorted(
                results_by_key.values(),
                key=lambda item: (
                    str(
                        item.get(
                            "query_id",
                            "",
                        )
                    ),
                    str(
                        item.get(
                            "condition",
                            "",
                        )
                    ),
                ),
            )

            write_jsonl(
                OUT_PATH,
                ordered_records,
            )

            print(
                "[OK] Judge berhasil."
                f" Total tersimpan: "
                f"{len(results_by_key)}/"
                f"{EXPECTED_RETRIEVAL_RECORDS}"
            )

        except Exception as exc:

            print(
                f"[ERROR] Gagal menilai "
                f"{query_id}/{condition}: {exc}"
            )

        # ----------------------------------------------------
        # REQUEST DELAY
        # ----------------------------------------------------

        if index < len(
            pending
        ):

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    ordered_records = sorted(
        results_by_key.values(),
        key=lambda item: (
            str(
                item.get(
                    "query_id",
                    "",
                )
            ),
            str(
                item.get(
                    "condition",
                    "",
                )
            ),
        ),
    )

    write_jsonl(
        OUT_PATH,
        ordered_records,
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )
    print(
        "LLM-AS-A-JUDGE SELESAI"
    )
    print(
        "=" * 60
    )

    print(
        f"Expected retrieval records : "
        f"{EXPECTED_RETRIEVAL_RECORDS}"
    )

    print(
        f"Judge records tersimpan    : "
        f"{len(results_by_key)}"
    )

    print(
        f"Missing records            : "
        f"{EXPECTED_RETRIEVAL_RECORDS - len(results_by_key)}"
    )

    print(
        f"Output                     : "
        f"{OUT_PATH}"
    )

    print(
        "=" * 60
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-a-Judge untuk evaluasi "
            "AniRAG-v2"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "calibration",
        ],
        default="full",
        help=(
            "Mode evaluasi: "
            "full atau calibration."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "Model Gemini yang digunakan "
            "sebagai judge."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    if args.mode == "calibration":

        run_calibration(
            judge_model=args.model
        )

    elif args.mode == "full":

        run_full(
            judge_model=args.model
        )


if __name__ == "__main__":
    main()