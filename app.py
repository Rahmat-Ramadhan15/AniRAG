"""
AniRAG-v2 — Gradio UI

UI Minimalis + Backend AniRAG-v2 + Output Format Terinterleave (Judul + Plot + Alasan + Poster)
"""

from __future__ import annotations

import html
import math
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr

# ============================================================
# PATH & BACKEND INITIALIZATION (AniRAG-v2)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

from src.rag_pipeline import RagPipeline

print("=" * 70)
print("AniRAG-v2 — INITIALIZING")
print("=" * 70)

pipe = RagPipeline()
if hasattr(pipe, "load_index"):
    pipe.load_index()

if os.environ.get("USE_GPU_BACKEND") == "1":
    print("[INFO] USE_GPU_BACKEND=1 -- memuat model GPU")
    if hasattr(pipe, "load_llm"):
        pipe.load_llm(quantize=True)

print("[OK] RagPipeline berhasil diinisialisasi.")


# ============================================================
# UI CONFIGURATION & CONSTANTS
# ============================================================

APP_TITLE = "AniRAG"
TOP_K = 5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
MIN_RELEVANCE_SCORE = 0.25

EXAMPLE_PROMPTS = [
    "Rekomendasikan anime action dengan tema samurai",
    "Aku suka Naruto Shippuden, ada anime yang mirip?",
    "Rekomendasikan anime sports dengan rating tinggi",
]

FOUND_HEADER = "✅ **Rekomendasi ditemukan!**\n\n"
NOT_FOUND_HEADER = "🔍 "

OUT_OF_DOMAIN_MESSAGE = (
    "Maaf, pertanyaan ini sepertinya di luar topik anime yang bisa saya bantu. "
    "Coba tanyakan rekomendasi anime, filter genre/tahun/rating, atau info seputar "
    "anime tertentu ya!"
)


# ============================================================
# HELPER FUNCTIONS & FORMATTING
# ============================================================

def _valid_score(value: Any) -> float | None:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _valid_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def parse_structured_answer(answer_text: str) -> list[dict[str, str]]:
    """
    Parse jawaban LLM berformat:
        ### Judul
        Plot: ...
        Alasan: ...
    """
    blocks = re.split(r"^###\s+", answer_text, flags=re.MULTILINE)
    parsed = []
    for block in blocks[1:]:
        lines = block.strip().split("\n", 1)
        title = lines[0].strip().strip("[]")
        body = lines[1].strip() if len(lines) > 1 else ""
        if title:
            parsed.append({"title": title, "body": body})
    return parsed


def match_to_retrieved(parsed_title: str, retrieved: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Cocokkan judul hasil parsing LLM ke metadata hasil retrieval.
    """
    title_lower = parsed_title.strip().lower()
    for d in retrieved:
        meta = d.get("metadata", d) if isinstance(d, dict) else {}
        candidates = [t for t in [meta.get("title"), meta.get("title_english")] if t]
        for c in candidates:
            c_lower = str(c).strip().lower()
            if c_lower == title_lower or c_lower in title_lower or title_lower in c_lower:
                return meta
    return None


def build_interleaved_message(answer_text: str, retrieved: list[dict[str, Any]]) -> str:
    """
    Menyusun format output terinterleave:
    Setiap '### Judul' diikuti Plot/Alasan, Gambar Poster, dan Badge Metadata.
    """
    parsed = parse_structured_answer(answer_text)
    if not parsed:
        return NOT_FOUND_HEADER + answer_text

    parts = [FOUND_HEADER.strip()]
    n_matched = 0

    for item in parsed:
        doc = match_to_retrieved(item["title"], retrieved)
        parts.append(f"### {item['title']}")
        
        if item["body"]:
            parts.append(item["body"])
            
        if doc and doc.get("image_url"):
            n_matched += 1
            meta_bits = []
            
            valid_score = _valid_score(doc.get("score") or doc.get("mal_score"))
            if valid_score is not None:
                meta_bits.append(f"★ {valid_score}")
                
            valid_genres = _valid_text(doc.get("genres"))
            if valid_genres:
                meta_bits.append(" | ".join(valid_genres.split("|")[:3]))
                
            valid_themes = _valid_text(doc.get("themes"))
            if valid_themes:
                meta_bits.append(" | ".join(valid_themes.split("|")[:2]))
                
            parts.append(f"![{item['title']}]({doc['image_url']})")
            if meta_bits:
                parts.append(f"_{' — '.join(meta_bits)}_")
                
        parts.append("---")

    if parts and parts[-1] == "---":
        parts.pop()

    if n_matched == 0:
        parts.append(
            "---\n\n⚠️ _Catatan: rekomendasi di atas tidak dapat diverifikasi dari "
            "basis data kami -- kemungkinan model menjawab dari pengetahuan umumnya "
            "sendiri, bukan dari data yang tersedia._"
        )

    return "\n\n".join(parts).strip()


# ============================================================
# HISTORY NORMALIZATION
# ============================================================

def normalize_history(history: Any) -> list[dict[str, Any]]:
    if not history:
        return []

    normalized = []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    normalized.append({"role": role, "content": content})
                continue

            if isinstance(item, (tuple, list)):
                if len(item) >= 1 and isinstance(item[0], str) and item[0].strip():
                    normalized.append({"role": "user", "content": item[0]})
                if len(item) >= 2 and isinstance(item[1], str) and item[1].strip():
                    normalized.append({"role": "assistant", "content": item[1]})

    return normalized


# ============================================================
# CHAT RESPONSE CALLBACK (AniRAG-v2)
# ============================================================

def respond(message: str, history: list[dict[str, Any]] | None):
    message = _valid_text(message)
    if not message:
        yield history or []
        return

    normalized_history = normalize_history(history)

    try:
        # Jalankan pipeline penuh AniRAG-v2
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
        print("\n" + "=" * 70)
        print("[ERROR] AniRAG generate()")
        print(repr(exc))
        print("=" * 70 + "\n")

        error_message = "Maaf, terjadi kesalahan saat memproses permintaan. Silakan coba lagi."
        new_history = list(history or [])
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": error_message})
        yield new_history
        return

    # Handle Blocked / Refusal
    if isinstance(result, dict) and result.get("blocked"):
        refusal = _valid_text(result.get("refusal") or result.get("response") or result.get("answer"))
        if not refusal:
            refusal = "Maaf, saya hanya dapat membantu pertanyaan yang berkaitan dengan anime."

        new_history = list(history or [])
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": refusal})
        yield new_history
        return

    # Ambil teks jawaban dan hasil retrieval dari result AniRAG-v2
    if isinstance(result, dict):
        raw_answer = _valid_text(result.get("response") or result.get("answer"))
        results_list = result.get("results", [])
    else:
        raw_answer = _valid_text(result)
        results_list = []

    if not isinstance(results_list, list):
        results_list = []

    # Format jawaban menjadi bentuk terinterleave (Markdown + Poster)
    final_message = build_interleaved_message(raw_answer, results_list)

    new_history = list(history or [])
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": final_message})

    yield new_history


# ============================================================
# TAMPILAN & CSS RESPONSIONAL
# ============================================================

EMPTY_STATE_HTML = (
    '<div id="empty-state">'
    '<div id="empty-state-icon">🤖</div>'
    '<div id="empty-state-title">Halo! Saya AniRAG</div>'
    '<div id="empty-state-subtitle">Tanyakan apa saja tentang anime — '
    'rekomendasi, episode, genre, skor, dan lainnya.</div>'
    '</div>'
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-page) !important;
    font-family: 'Inter', -apple-system, "Segoe UI", Roboto, sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 16px !important;
}

html, body { height: 100%; margin: 0 !important; overflow: hidden !important; }
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    height: 100dvh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
.gradio-container > .main, .gradio-container > div:first-child {
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
#chatbot .wrap, #chatbot .bubble-wrap {
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
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
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

.message-wrap { padding: 6px 4px !important; }
.message, .message p, .message li {
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
.message.user, .message.user * {
    color: #FFFFFF !important;
}
.message.bot {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: var(--text-primary) !important;
}
.message.bot h3, .message.bot strong { color: var(--text-primary); }

#bottom-dock {
    flex: 0 0 auto !important;
    background: var(--bg-surface);
    border-top: 1px solid var(--border);
    padding: 12px 20px !important;
    padding-bottom: max(12px, env(safe-area-inset-bottom)) !important;
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
#input-row textarea, #input-row input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
    padding: 12px 18px !important;
    min-height: 0 !important;
}
#input-row textarea::placeholder, #input-row input::placeholder {
    color: var(--text-muted) !important;
}
#input-row textarea:focus, #input-row input:focus {
    border-color: var(--border-strong) !important;
    box-shadow: 0 0 0 3px rgba(17, 17, 20, 0.06) !important;
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

footer { display: none !important; }
button[aria-label="Settings"],
button[aria-label*="hortcut" i],
button[aria-label*="elp" i],
button[title*="hortcut" i],
button[title*="elp" i] {
    display: none !important;
}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-page); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #B5B5BC; }

@media (max-width: 640px) {
    body, .gradio-container { font-size: 15px !important; }
    #app-header { padding: 14px 16px; }
    #header-title { font-size: 16px; }
    #app-header p { display: none !important; }
    #chat-scroll { padding: 12px 14px 0 14px; }
    #empty-state { padding: 24px 16px; }
    #empty-state-icon { width: 48px; height: 48px; font-size: 20px; border-radius: 14px; }
    #empty-state-title { font-size: 16px; }
    #empty-state-subtitle { font-size: 13px; }
    .message, .message p, .message li { font-size: 14.5px !important; line-height: 1.55 !important; }
    .message h3 { font-size: 15px !important; margin-top: 10px !important; }
    #bottom-dock { padding: 10px 14px !important; padding-bottom: max(10px, env(safe-area-inset-bottom)) !important; gap: 8px !important; }
    .example-btn { font-size: 12.5px !important; padding: 6px 12px !important; }
    #input-row {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 6px !important;
    }
    #input-row > *:first-child { flex: 1 1 auto !important; }
    #input-row > *:last-child { flex: 0 0 44px !important; min-width: 0 !important; }
    #input-row textarea, #input-row input {
        font-size: 14.5px !important;
        padding: 10px 14px !important;
    }
}
"""

CLEANUP_JS = """
() => {
    function hideFloatingHelpButton() {
        document.querySelectorAll('body *').forEach((el) => {
            const style = window.getComputedStyle(el);
            if (style.position !== 'fixed') return;
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return;
            if (rect.width > 60 || rect.height > 60) return;
            const nearRight = window.innerWidth - rect.right < 60;
            const nearBottom = window.innerHeight - rect.bottom < 60;
            if (nearRight && nearBottom) {
                el.style.setProperty('display', 'none', 'important');
            }
        });
    }
    hideFloatingHelpButton();
    new MutationObserver(hideFloatingHelpButton).observe(document.body, { childList: true, subtree: true });
}
"""

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Inter"), "monospace"],
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
# GRADIO APPLICATION BUILD
# ============================================================

with gr.Blocks(title=APP_TITLE) as demo:
    with gr.Column(elem_id="page-wrap"):
        gr.HTML(
            '<div id="app-header">'
            '<span id="header-title">AniRAG</span>'
            '<p>Chatbot rekomendasi anime berbasis RAG dan SLM yang dilengkapi '
            'dengan poster anime.</p>'
            '</div>'
        )

        with gr.Column(elem_id="chat-scroll"):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                height="100%",
                label="Percakapan",
                show_label=False,
                avatar_images=None,
                placeholder=EMPTY_STATE_HTML,
            )

        with gr.Column(elem_id="bottom-dock"):
            with gr.Row(elem_id="examples-row") as examples_row:
                example_buttons = [
                    gr.Button(p, elem_classes="example-btn", size="sm")
                    for p in EXAMPLE_PROMPTS
                ]

            with gr.Row(elem_id="input-row"):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Tanyakan sesuatu tentang anime...",
                    scale=8,
                    lines=1,
                    autofocus=True,
                    container=False,
                )
                send_btn = gr.Button("➤", elem_id="send-btn", scale=1, variant="primary")

            gr.HTML('<p id="disclaimer">AniRAG dapat membuat kesalahan. Periksa informasi penting.</p>')

    for btn, prompt_text in zip(example_buttons, EXAMPLE_PROMPTS):
        btn.click(fn=lambda p=prompt_text: p, inputs=None, outputs=msg)

    msg.submit(
        respond, [msg, chatbot], [chatbot]
    ).then(
        lambda: "", None, msg
    ).then(
        lambda: gr.update(visible=False), None, examples_row
    )

    send_btn.click(
        respond, [msg, chatbot], [chatbot]
    ).then(
        lambda: "", None, msg
    ).then(
        lambda: gr.update(visible=False), None, examples_row
    )


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AniRAG-v2 — Gradio UI")
    print("=" * 70)
    print("[INFO] Menjalankan aplikasi...")

    share_mode = os.environ.get("GRADIO_SHARE") == "1"
    demo.launch(theme=theme, css=CUSTOM_CSS, js=CLEANUP_JS, share=share_mode)
