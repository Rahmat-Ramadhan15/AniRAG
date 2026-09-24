"""
AniRAG-v2 — Gradio UI

UI minimalis untuk AniRAG-v2.

Alur:
User Query
    ↓
RagPipeline
    ↓
Retrieval + Filtering + Reranking
    ↓
SLM
    ↓
Jawaban
    ↓
UI + Metadata Poster

Catatan:
- Poster diambil dari metadata retrieval, bukan dari output LLM.
- TOP_K adalah jumlah kandidat retrieval, bukan jumlah rekomendasi.
- Jumlah rekomendasi ditentukan oleh instruksi pada SYSTEM_PROMPT_RAG.
"""

from __future__ import annotations

import html
import math
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr

from src.rag_pipeline import RagPipeline


# ============================================================
# PATH & BACKEND INITIALIZATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

print("=" * 70)
print("AniRAG-v2 — INITIALIZING")
print("=" * 70)

print(f"[INFO] PROJECT_ROOT : {PROJECT_ROOT}")
print(f"[INFO] Working dir  : {Path.cwd()}")

# ------------------------------------------------------------
# Initialize RAG pipeline
# ------------------------------------------------------------

pipe = RagPipeline()

# Load FAISS index if supported by current pipeline.
if hasattr(pipe, "load_index"):
    print("[INFO] Memuat FAISS index...")
    pipe.load_index()

# ------------------------------------------------------------
# GPU backend
# ------------------------------------------------------------

USE_GPU_BACKEND = os.environ.get("USE_GPU_BACKEND", "0") == "1"

if USE_GPU_BACKEND:
    print("[INFO] USE_GPU_BACKEND=1 -- memuat model GPU")

    if hasattr(pipe, "load_llm"):
        pipe.load_llm(quantize=True)
    else:
        print("[WARNING] RagPipeline tidak memiliki method load_llm().")

else:
    print("[INFO] USE_GPU_BACKEND=0 -- model LLM tidak dimuat saat startup.")

print("[OK] RagPipeline berhasil diinisialisasi.")
print("=" * 70)


# ============================================================
# UI CONFIGURATION
# ============================================================

APP_TITLE = "AniRAG"

# Jumlah kandidat yang diberikan retrieval kepada pipeline.
# Ini BUKAN jumlah rekomendasi yang wajib ditampilkan.
TOP_K = 5

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2


EXAMPLE_PROMPTS = [
    "Rekomendasikan anime action dengan tema samurai",
    "Aku suka Naruto Shippuden, ada anime yang mirip?",
    "Rekomendasikan anime sports dengan rating tinggi",
]


FOUND_HEADER = "✅ **Rekomendasi ditemukan!**"

NOT_FOUND_HEADER = "🔍 "

DEFAULT_REFUSAL_MESSAGE = (
    "Maaf, saya hanya dapat membantu pertanyaan "
    "yang berkaitan dengan anime."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _valid_score(value: Any) -> float | None:
    """
    Mengubah nilai score menjadi float jika valid.
    """
    try:
        if value is None:
            return None

        f = float(value)

        if math.isnan(f):
            return None

        return f

    except (TypeError, ValueError):
        return None


def _valid_text(value: Any) -> str | None:
    """
    Membersihkan text sederhana.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.lower() == "nan":
        return None

    return text


def _escape(value: Any) -> str:
    """
    HTML escaping untuk metadata yang akan dimasukkan
    ke HTML.
    """
    text = _valid_text(value)

    if text is None:
        return ""

    return html.escape(text)


# ============================================================
# TITLE NORMALIZATION
# ============================================================

def normalize_title(title: Any) -> str:
    """
    Normalisasi judul untuk pencocokan.

    Tidak mengubah metadata asli.
    Hanya digunakan untuk matching.
    """
    text = _valid_text(title)

    if not text:
        return ""

    text = text.lower().strip()

    # Hilangkan markdown sederhana.
    text = re.sub(r"[*_`~]", "", text)

    # Hilangkan punctuation.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalisasi whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# PARSE STRUCTURED LLM RESPONSE
# ============================================================

def parse_structured_answer(
    answer_text: str | None,
) -> list[dict[str, str]]:
    """
    Parse output LLM menjadi:

    [
        {
            "title": "...",
            "body": "..."
        }
    ]

    Format utama:

    ### Judul
    Plot: ...
    Alasan: ...

    Fallback juga menangani beberapa variasi sederhana.
    """

    answer_text = _valid_text(answer_text)

    if not answer_text:
        return []

    parsed: list[dict[str, str]] = []

    # --------------------------------------------------------
    # FORMAT 1
    #
    # ### Judul
    # Plot: ...
    # Alasan: ...
    # --------------------------------------------------------

    if re.search(
        r"^###\s+",
        answer_text,
        flags=re.MULTILINE,
    ):
        blocks = re.split(
            r"^###\s+",
            answer_text.strip(),
            flags=re.MULTILINE,
        )

        for block in blocks[1:]:
            block = block.strip()

            if not block:
                continue

            lines = block.splitlines()

            if not lines:
                continue

            title = lines[0].strip()

            # Bersihkan markdown.
            title = title.strip("*[]` ")

            body_lines = lines[1:]

            body = "\n".join(
                line.strip()
                for line in body_lines
                if line.strip()
            ).strip()

            # ------------------------------------------------
            # Jika judul masih mengandung "Plot:"
            # ------------------------------------------------

            inline_match = re.match(
                r"^(.*?)\s+Plot\s*:\s*(.*)$",
                title,
                flags=re.IGNORECASE,
            )

            if inline_match:
                title = inline_match.group(1).strip()

                inline_plot = inline_match.group(2).strip()

                if inline_plot:
                    body = (
                        f"Plot: {inline_plot}\n"
                        f"{body}"
                    ).strip()

            if title:
                parsed.append(
                    {
                        "title": title,
                        "body": body,
                    }
                )

        if parsed:
            return parsed

    # --------------------------------------------------------
    # FORMAT 2
    #
    # 1. Judul - Plot: ...
    #
    # atau
    #
    # - Judul: ...
    # --------------------------------------------------------

    lines = answer_text.splitlines()

    for line in lines:

        line_item = line.strip()

        if not line_item:
            continue

        # Hilangkan bullet/list.
        line_item = re.sub(
            r"^(?:\d+[\.\)]|[-*•])\s*",
            "",
            line_item,
        ).strip()

        match = re.match(
            r"^(.*?)\s*[:\-—]\s*(.+)$",
            line_item,
        )

        if not match:
            continue

        title = match.group(1).strip()
        body = match.group(2).strip()

        title = title.strip("*[]` ")

        # Jangan menganggap kalimat biasa sebagai judul.
        if len(title) < 3:
            continue

        excluded_titles = {
            "berikut",
            "catatan",
            "semoga",
            "rekomendasi",
            "plot",
            "alasan",
        }

        if title.lower() in excluded_titles:
            continue

        title = re.sub(
            r"\s+Plot$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()

        parsed.append(
            {
                "title": title,
                "body": body,
            }
        )

    return parsed


# ============================================================
# MATCH LLM TITLE TO RETRIEVED METADATA
# ============================================================

def match_to_retrieved(
    parsed_title: str,
    retrieved: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Mencocokkan judul dari output LLM dengan metadata
    hasil retrieval.

    Prioritas:
    1. Exact title
    2. Exact English title
    3. Substring yang aman

    Return:
        metadata dictionary
        atau None jika tidak ditemukan.
    """

    normalized_query_title = normalize_title(parsed_title)

    if not normalized_query_title:
        return None

    # --------------------------------------------------------
    # FIRST PASS: exact matching
    # --------------------------------------------------------

    for item in retrieved:

        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata", item)

        if not isinstance(metadata, dict):
            continue

        candidates = [
            metadata.get("title"),
            metadata.get("title_english"),
        ]

        for candidate in candidates:

            candidate_normalized = normalize_title(candidate)

            if not candidate_normalized:
                continue

            if candidate_normalized == normalized_query_title:
                return metadata

    # --------------------------------------------------------
    # SECOND PASS: substring matching
    # --------------------------------------------------------

    for item in retrieved:

        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata", item)

        if not isinstance(metadata, dict):
            continue

        candidates = [
            metadata.get("title"),
            metadata.get("title_english"),
        ]

        for candidate in candidates:

            candidate_normalized = normalize_title(candidate)

            if not candidate_normalized:
                continue

            if (
                candidate_normalized in normalized_query_title
                or normalized_query_title in candidate_normalized
            ):
                return metadata

    return None


# ============================================================
# BUILD METADATA LINE
# ============================================================

def build_metadata_line(
    metadata: dict[str, Any],
) -> str:
    """
    Membuat informasi metadata dari dataset.

    Contoh:
    ★ 8.72 — Action — Samurai
    """

    meta_bits: list[str] = []

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = _valid_score(
        metadata.get("score")
        or metadata.get("mal_score")
    )

    if score is not None:
        meta_bits.append(f"★ {score:g}")

    # --------------------------------------------------------
    # Genres
    # --------------------------------------------------------

    genres = _valid_text(metadata.get("genres"))

    if genres:
        genre_items = [
            item.strip()
            for item in genres.split("|")
            if item.strip()
        ]

        if genre_items:
            meta_bits.append(
                " • ".join(genre_items[:3])
            )

    # --------------------------------------------------------
    # Themes
    # --------------------------------------------------------

    themes = _valid_text(metadata.get("themes"))

    if themes:
        theme_items = [
            item.strip()
            for item in themes.split("|")
            if item.strip()
        ]

        if theme_items:
            meta_bits.append(
                " • ".join(theme_items[:2])
            )

    if not meta_bits:
        return ""

    return " — ".join(meta_bits)


# ============================================================
# BUILD ANIME CARD
# ============================================================

def build_anime_card(
    title: str,
    body: str,
    metadata: dict[str, Any] | None,
) -> str:
    """
    Membuat satu blok rekomendasi.

    Poster berasal dari metadata.image_url.
    """

    parts: list[str] = []

    safe_title = _valid_text(title) or "Anime"

    parts.append(
        f"### {safe_title}"
    )

    if body:
        parts.append(body)

    if metadata:

        image_url = _valid_text(
            metadata.get("image_url")
        )

        if image_url:

            safe_url = html.escape(
                image_url,
                quote=True,
            )

            safe_alt = html.escape(
                safe_title,
                quote=True,
            )

            parts.append(
                f"![{safe_alt}]({safe_url})"
            )

        metadata_line = build_metadata_line(
            metadata
        )

        if metadata_line:
            parts.append(
                f"_{metadata_line}_"
            )

    return "\n\n".join(parts)


# ============================================================
# BUILD FINAL INTERLEAVED RESPONSE
# ============================================================

def build_interleaved_message(
    answer_text: str | None,
    retrieved: list[dict[str, Any]],
) -> str:
    """
    Menggabungkan jawaban LLM dengan metadata retrieval.

    Poster tidak berasal dari LLM.
    Poster berasal dari metadata.image_url.
    """

    answer_text = _valid_text(answer_text)

    if not answer_text:
        return (
            "Maaf, saya tidak mendapatkan jawaban "
            "yang dapat ditampilkan."
        )

    parsed = parse_structured_answer(
        answer_text
    )

    # --------------------------------------------------------
    # Jika LLM menjawab factual question atau format biasa.
    # Jangan paksa menjadi recommendation card.
    # --------------------------------------------------------

    if not parsed:
        return (
            f"{NOT_FOUND_HEADER}"
            f"{answer_text}"
        )

    parts: list[str] = [
        FOUND_HEADER
    ]

    matched_count = 0

    for item in parsed:

        title = item.get("title", "")
        body = item.get("body", "")

        metadata = match_to_retrieved(
            title,
            retrieved,
        )

        if metadata:
            matched_count += 1

        card = build_anime_card(
            title=title,
            body=body,
            metadata=metadata,
        )

        parts.append(card)

    # --------------------------------------------------------
    # Separator antar recommendation.
    # --------------------------------------------------------

    return "\n\n---\n\n".join(parts).strip()


# ============================================================
# HISTORY NORMALIZATION
# ============================================================

def normalize_history(
    history: Any,
) -> list[dict[str, str]]:
    """
    Menormalisasi history Gradio menjadi:

    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]

    Mendukung:
    - Gradio messages format
    - legacy tuple format
    """

    if not history:
        return []

    normalized: list[dict[str, str]] = []

    if not isinstance(history, list):
        return normalized

    for item in history:

        # ----------------------------------------------------
        # Gradio messages format
        # ----------------------------------------------------

        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if (
                role in {"user", "assistant"}
                and isinstance(content, str)
                and content.strip()
            ):
                normalized.append(
                    {
                        "role": role,
                        "content": content.strip(),
                    }
                )

            continue

        # ----------------------------------------------------
        # Legacy tuple/list format
        # ----------------------------------------------------

        if isinstance(item, (tuple, list)):

            if (
                len(item) >= 1
                and isinstance(item[0], str)
                and item[0].strip()
            ):
                normalized.append(
                    {
                        "role": "user",
                        "content": item[0].strip(),
                    }
                )

            if (
                len(item) >= 2
                and isinstance(item[1], str)
                and item[1].strip()
            ):
                normalized.append(
                    {
                        "role": "assistant",
                        "content": item[1].strip(),
                    }
                )

    return normalized


# ============================================================
# CHAT RESPONSE CALLBACK
# ============================================================

def respond(
    message: str,
    history: list[dict[str, Any]] | None,
):
    """
    Callback utama Gradio.
    """

    message = _valid_text(message)

    if not message:
        yield history or []
        return

    normalized_history = normalize_history(
        history
    )

    try:

        # ----------------------------------------------------
        # Jalankan pipeline AniRAG-v2
        # ----------------------------------------------------

        result = pipe.generate(
            message,
            history=normalized_history,
            top_k=TOP_K,
            use_retrieval=True,
            use_enrichment=False,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("[ERROR] AniRAG generate()")
        print(repr(exc))
        print("=" * 70)
        print()

        error_message = (
            "Maaf, terjadi kesalahan saat memproses "
            "permintaan. Silakan coba lagi."
        )

        new_history = list(
            history or []
        )

        new_history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        new_history.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )

        yield new_history
        return

    # ========================================================
    # BLOCKED / REFUSAL
    # ========================================================

    if (
        isinstance(result, dict)
        and result.get("blocked")
    ):

        refusal = _valid_text(
            result.get("refusal")
            or result.get("response")
            or result.get("answer")
        )

        if not refusal:
            refusal = DEFAULT_REFUSAL_MESSAGE

        new_history = list(
            history or []
        )

        new_history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        new_history.append(
            {
                "role": "assistant",
                "content": refusal,
            }
        )

        yield new_history
        return

    # ========================================================
    # EXTRACT RESULT
    # ========================================================

    if isinstance(result, dict):

        raw_answer = _valid_text(
            result.get("response")
            or result.get("answer")
        )

        results_list = result.get(
            "results",
            [],
        )

    else:

        raw_answer = _valid_text(
            result
        )

        results_list = []

    if not isinstance(
        results_list,
        list,
    ):
        results_list = []

    # ========================================================
    # DEBUG
    # ========================================================
    #
    # Sengaja ditampilkan di Kaggle console agar mudah
    # membedakan:
    #
    # 1. masalah LLM
    # 2. masalah retrieval
    # 3. masalah parser/UI
    #
    # ========================================================

    print()
    print("=" * 70)
    print("[DEBUG] USER QUERY")
    print(message)

    print("-" * 70)
    print("[DEBUG] RAW LLM RESPONSE")
    print(raw_answer)

    print("-" * 70)
    print("[DEBUG] RETRIEVED RESULT COUNT")
    print(len(results_list))

    print("-" * 70)
    print("[DEBUG] RETRIEVED TITLES")

    for index, item in enumerate(
        results_list,
        start=1,
    ):

        if not isinstance(item, dict):
            continue

        metadata = item.get(
            "metadata",
            item,
        )

        if not isinstance(
            metadata,
            dict,
        ):
            continue

        title = (
            metadata.get("title")
            or metadata.get("title_english")
            or "Unknown"
        )

        print(
            f"{index}. {title}"
        )

    print("=" * 70)
    print()

    # ========================================================
    # FORMAT FINAL RESPONSE
    # ========================================================

    final_message = build_interleaved_message(
        raw_answer,
        results_list,
    )

    # ========================================================
    # UPDATE CHAT HISTORY
    # ========================================================

    new_history = list(
        history or []
    )

    new_history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    new_history.append(
        {
            "role": "assistant",
            "content": final_message,
        }
    )

    yield new_history


# ============================================================
# EMPTY STATE
# ============================================================

EMPTY_STATE_HTML = (
    '<div id="empty-state">'
    '<div id="empty-state-icon">🤖</div>'
    '<div id="empty-state-title">'
    'Halo! Saya AniRAG'
    '</div>'
    '<div id="empty-state-subtitle">'
    'Tanyakan apa saja tentang anime — '
    'rekomendasi, episode, genre, skor, dan lainnya.'
    '</div>'
    '</div>'
)


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
);

:root {
    --bg-page: #F7F7F8;
    --bg-surface: #FFFFFF;
    --border: #E3E3E6;
    --border-strong: #D0D0D5;
    --text-primary: #111114;
    --text-secondary: #6B6F76;
    --text-muted: #9AA0A6;
    --accent: #111114;
    --accent-soft: #F0F0F2;
}

* {
    box-sizing: border-box;
}

body,
.gradio-container {
    background: var(--bg-page) !important;
    font-family:
        'Inter',
        -apple-system,
        "Segoe UI",
        Roboto,
        sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 16px !important;
}

html,
body {
    height: 100%;
    margin: 0 !important;
    overflow: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    height: 100dvh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.gradio-container > .main,
.gradio-container > div:first-child {
    height: 100% !important;
}

#page-wrap {
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    height: 100dvh !important;
    max-height: 100vh !important;
    max-height: 100dvh !important;
    padding: 0 !important;
    gap: 0 !important;
}

#app-header {
    display: flex !important;
    align-items: flex-start !important;
    justify-content: space-between !important;
    gap: 20px;
    flex: 0 0 auto;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    padding: 18px 24px;
}

#header-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    white-space: nowrap;
    line-height: 1.3;
}

#app-header p {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    text-align: right;
    max-width: 320px;
}

#chat-scroll {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--bg-page);
    padding: 16px 20px 0 20px;
}

#chatbot {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#chatbot .wrap,
#chatbot .bubble-wrap {
    overflow-y: auto !important;
    -webkit-overflow-scrolling: touch !important;
}

#empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 6px;
    height: 100%;
    padding: 40px 20px;
}

#empty-state-icon {
    width: 56px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    margin-bottom: 8px;
}

#empty-state-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
}

#empty-state-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    max-width: 360px;
    line-height: 1.5;
}

.message-wrap {
    padding: 6px 4px !important;
}

.message,
.message p,
.message li {
    font-size: 15px !important;
    line-height: 1.6 !important;
    word-break: break-word !important;
}

.message h3 {
    font-family: 'Inter', sans-serif;
    font-size: 16px !important;
    font-weight: 600;
    margin-top: 12px !important;
}

.message img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    max-width: 200px !important;
    width: 100% !important;
    height: auto !important;
    margin-top: 8px !important;
    box-shadow: none !important;
}

.message.user {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 18px 18px 4px 18px !important;
}

.message.user,
.message.user * {
    color: #FFFFFF !important;
}

.message.bot {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: var(--text-primary) !important;
}

.message.bot h3,
.message.bot strong {
    color: var(--text-primary);
}

#bottom-dock {
    flex: 0 0 auto !important;
    background: var(--bg-surface);
    border-top: 1px solid var(--border);
    padding: 12px 20px !important;
    padding-bottom: max(
        12px,
        env(safe-area-inset-bottom)
    ) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
}

#examples-row {
    flex-wrap: wrap !important;
    gap: 8px !important;
}

.example-btn {
    border: 1px solid var(--border) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    min-height: 0 !important;
    white-space: nowrap !important;
    flex: 0 0 auto !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}

.example-btn:hover {
    border-color: var(--border-strong) !important;
    background: var(--accent-soft) !important;
}

#input-row {
    align-items: center !important;
    gap: 8px !important;
}

#input-row textarea,
#input-row input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
    padding: 12px 18px !important;
    min-height: 0 !important;
}

#input-row textarea::placeholder,
#input-row input::placeholder {
    color: var(--text-muted) !important;
}

#input-row textarea:focus,
#input-row input:focus {
    border-color: var(--border-strong) !important;
    box-shadow:
        0 0 0 3px
        rgba(17, 17, 20, 0.06) !important;
}

#send-btn {
    background: var(--accent-soft) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    flex: 0 0 44px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
}

#send-btn:hover {
    background: var(--border) !important;
}

#disclaimer {
    margin: 0 !important;
    text-align: center !important;
    font-size: 11.5px !important;
    color: var(--text-muted) !important;
}

footer {
    display: none !important;
}

button[aria-label="Settings"],
button[aria-label*="shortcut" i],
button[aria-label*="help" i],
button[title*="shortcut" i],
button[title*="help" i] {
    display: none !important;
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-page);
}

::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 8px;
}

::-webkit-scrollbar-thumb:hover {
    background: #B5B5BC;
}

@media (max-width: 640px) {

    body,
    .gradio-container {
        font-size: 15px !important;
    }

    #app-header {
        padding: 14px 16px;
    }

    #header-title {
        font-size: 16px;
    }

    #app-header p {
        display: none !important;
    }

    #chat-scroll {
        padding: 12px 14px 0 14px;
    }

    #empty-state {
        padding: 24px 16px;
    }

    #empty-state-icon {
        width: 48px;
        height: 48px;
        font-size: 20px;
        border-radius: 14px;
    }

    #empty-state-title {
        font-size: 16px;
    }

    #empty-state-subtitle {
        font-size: 13px;
    }

    .message,
    .message p,
    .message li {
        font-size: 14.5px !important;
        line-height: 1.55 !important;
    }

    .message h3 {
        font-size: 15px !important;
        margin-top: 10px !important;
    }

    #bottom-dock {
        padding: 10px 14px !important;
        padding-bottom:
            max(
                10px,
                env(safe-area-inset-bottom)
            ) !important;
        gap: 8px !important;
    }

    .example-btn {
        font-size: 12.5px !important;
        padding: 6px 12px !important;
    }

    #input-row {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 6px !important;
    }

    #input-row > *:first-child {
        flex: 1 1 auto !important;
    }

    #input-row > *:last-child {
        flex: 0 0 44px !important;
        min-width: 0 !important;
    }

    #input-row textarea,
    #input-row input {
        font-size: 14.5px !important;
        padding: 10px 14px !important;
    }
}
"""


# ============================================================
# JAVASCRIPT CLEANUP
# ============================================================

CLEANUP_JS = """
() => {

    function hideFloatingHelpButton() {

        document
            .querySelectorAll('body *')
            .forEach((el) => {

                const style =
                    window.getComputedStyle(el);

                if (style.position !== 'fixed') {
                    return;
                }

                const rect =
                    el.getBoundingClientRect();

                if (
                    rect.width === 0 ||
                    rect.height === 0
                ) {
                    return;
                }

                if (
                    rect.width > 60 ||
                    rect.height > 60
                ) {
                    return;
                }

                const nearRight =
                    window.innerWidth -
                    rect.right < 60;

                const nearBottom =
                    window.innerHeight -
                    rect.bottom < 60;

                if (
                    nearRight &&
                    nearBottom
                ) {
                    el.style.setProperty(
                        'display',
                        'none',
                        'important'
                    );
                }
            });
    }

    hideFloatingHelpButton();

    new MutationObserver(
        hideFloatingHelpButton
    ).observe(
        document.body,
        {
            childList: true,
            subtree: true
        }
    );
}
"""


# ============================================================
# GRADIO THEME
# ============================================================

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Inter"),
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("Inter"),
        "monospace",
    ],
).set(
    body_background_fill="#F7F7F8",
    body_background_fill_dark="#F7F7F8",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#E3E3E6",
    block_border_color_dark="#E3E3E6",
    body_text_color="#111114",
    body_text_color_dark="#111114",
    body_text_size="16px",
    button_large_text_size="16px",
)


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title=APP_TITLE,
    theme=theme,
    css=CUSTOM_CSS,
    js=CLEANUP_JS,
) as demo:

    with gr.Column(
        elem_id="page-wrap"
    ):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        gr.HTML(
            """
            <div id="app-header">
                <span id="header-title">AniRAG</span>

                <p>
                    Chatbot rekomendasi anime berbasis
                    RAG dan SLM yang dilengkapi
                    dengan poster anime.
                </p>
            </div>
            """
        )

        # ----------------------------------------------------
        # CHAT
        # ----------------------------------------------------

        with gr.Column(
            elem_id="chat-scroll"
        ):

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                height="100%",
                label="Percakapan",
                show_label=False,
                avatar_images=None,
                placeholder=EMPTY_STATE_HTML,
            )

        # ----------------------------------------------------
        # BOTTOM DOCK
        # ----------------------------------------------------

        with gr.Column(
            elem_id="bottom-dock"
        ):

            with gr.Row(
                elem_id="examples-row"
            ) as examples_row:

                example_buttons = [
                    gr.Button(
                        prompt,
                        elem_classes="example-btn",
                        size="sm",
                    )
                    for prompt in EXAMPLE_PROMPTS
                ]

            with gr.Row(
                elem_id="input-row"
            ):

                msg = gr.Textbox(
                    show_label=False,
                    placeholder=(
                        "Tanyakan sesuatu tentang anime..."
                    ),
                    scale=8,
                    lines=1,
                    autofocus=True,
                    container=False,
                )

                send_btn = gr.Button(
                    "➤",
                    elem_id="send-btn",
                    scale=1,
                    variant="primary",
                )

            gr.HTML(
                """
                <p id="disclaimer">
                    AniRAG dapat membuat kesalahan.
                    Periksa informasi penting.
                </p>
                """
            )

    # ========================================================
    # EXAMPLE BUTTONS
    # ========================================================

    for btn, prompt_text in zip(
        example_buttons,
        EXAMPLE_PROMPTS,
    ):

        btn.click(
            fn=lambda p=prompt_text: p,
            inputs=None,
            outputs=msg,
        )

    # ========================================================
    # SUBMIT VIA ENTER
    # ========================================================

    msg.submit(
        fn=respond,
        inputs=[
            msg,
            chatbot,
        ],
        outputs=[
            chatbot,
        ],
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=msg,
    ).then(
        fn=lambda: gr.update(
            visible=False
        ),
        inputs=None,
        outputs=examples_row,
    )

    # ========================================================
    # SUBMIT VIA SEND BUTTON
    # ========================================================

    send_btn.click(
        fn=respond,
        inputs=[
            msg,
            chatbot,
        ],
        outputs=[
            chatbot,
        ],
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=msg,
    ).then(
        fn=lambda: gr.update(
            visible=False
        ),
        inputs=None,
        outputs=examples_row,
    )


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AniRAG-v2 — Gradio UI")
    print("=" * 70)
    print("[INFO] Menjalankan aplikasi...")

    # --------------------------------------------------------
    # Kaggle:
    #
    # USE_GPU_BACKEND=1
    # GRADIO_SHARE=1
    #
    # Contoh:
    # os.environ["USE_GPU_BACKEND"] = "1"
    # os.environ["GRADIO_SHARE"] = "1"
    # --------------------------------------------------------

    share_mode = (
        os.environ.get(
            "GRADIO_SHARE",
            "0",
        ) == "1"
    )

    print(
        f"[INFO] GRADIO_SHARE={share_mode}"
    )

    demo.launch(
        share=share_mode,
    )
