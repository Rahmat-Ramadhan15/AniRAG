"""
AniRAG-v2 — Gradio UI (Desain Minimalis UI Acuan + Backend AniRAG-v2)

Entry point utama aplikasi.
UI mengambil desain penuh dari versi referensi (full-bleed shell, header flat, 
bottom-dock, chip contoh, empty state kustom, responsif mobile/desktop),
sedangkan seluruh alur pemrosesan data murni menggunakan RagPipeline AniRAG-v2.
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import gradio as gr

# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# BACKEND INITIALIZATION (AniRAG-v2)
# ============================================================

from src.rag_pipeline import RagPipeline

print("=" * 70)
print("AniRAG-v2 — INITIALIZING")
print("=" * 70)

pipe = RagPipeline()

print("[OK] RagPipeline berhasil diinisialisasi.")


# ============================================================
# UI CONFIGURATION
# ============================================================

APP_TITLE = "AniRAG"
TOP_K = 5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

EXAMPLE_PROMPTS = [
    "Rekomendasikan anime action dengan tema samurai",
    "Aku suka Naruto Shippuden, ada anime yang mirip?",
    "Rekomendasikan anime sports dengan rating tinggi",
]


# ============================================================
# HELPER FUNCTIONS & METADATA (AniRAG-v2)
# ============================================================

def _valid_text(value: Any) -> str:
    """Mengubah nilai menjadi string yang aman untuk ditampilkan."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _valid_score(value: Any) -> str:
    """Format score untuk UI."""
    if value is None:
        return ""
    try:
        score = float(value)
        if score <= 0:
            return ""
        return f"{score:.2f}"
    except (TypeError, ValueError):
        return _valid_text(value)


def _metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Mengambil metadata dari satu hasil retrieval."""
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return metadata


def _escape(value: Any) -> str:
    """HTML escape untuk mencegah metadata merusak markup UI."""
    return html.escape(_valid_text(value))


# ============================================================
# RESPONSE FORMATTING (AniRAG-v2 Cards)
# ============================================================

def build_anime_card(result: dict[str, Any]) -> str:
    """
    Membuat satu kartu anime berdasarkan metadata retrieval.
    Poster WAJIB berasal dari metadata.image_url.
    Tidak pernah mengambil URL poster dari output LLM.
    """
    metadata = _metadata(result)

    title = _valid_text(metadata.get("title"))
    title_english = _valid_text(metadata.get("title_english"))
    anime_type = _valid_text(metadata.get("type"))
    episodes = _valid_text(metadata.get("episodes"))
    score = _valid_score(metadata.get("score"))
    genres = _valid_text(metadata.get("genres"))
    themes = _valid_text(metadata.get("themes"))
    year = _valid_text(metadata.get("year"))
    studios = _valid_text(metadata.get("studios"))
    image_url = _valid_text(metadata.get("image_url"))

    # Poster
    if image_url:
        poster_html = f"""
        <img
            src="{_escape(image_url)}"
            alt="{_escape(title)}"
            class="anime-poster"
            loading="lazy"
        />
        """
    else:
        poster_html = """
        <div class="anime-poster anime-poster-empty">
            <span>No Poster</span>
        </div>
        """

    # Metadata
    meta_items = []
    if anime_type:
        meta_items.append(f'<span class="anime-meta-item">{_escape(anime_type)}</span>')
    if episodes:
        meta_items.append(f'<span class="anime-meta-item">{_escape(episodes)} eps</span>')
    if year:
        meta_items.append(f'<span class="anime-meta-item">{_escape(year)}</span>')
    if score:
        meta_items.append(f'<span class="anime-score">★ {_escape(score)}</span>')

    meta_html = "".join(meta_items)

    # Optional fields
    english_html = ""
    if title_english and title_english.lower() != title.lower():
        english_html = f'<div class="anime-english-title">{_escape(title_english)}</div>'

    genres_html = ""
    if genres:
        genres_html = f"""
        <div class="anime-detail">
            <span class="anime-label">Genre</span>
            <span>{_escape(genres)}</span>
        </div>
        """

    themes_html = ""
    if themes:
        themes_html = f"""
        <div class="anime-detail">
            <span class="anime-label">Tema</span>
            <span>{_escape(themes)}</span>
        </div>
        """

    studios_html = ""
    if studios:
        studios_html = f"""
        <div class="anime-detail">
            <span class="anime-label">Studio</span>
            <span>{_escape(studios)}</span>
        </div>
        """

    return f"""
    <div class="anime-card">
        <div class="anime-card-poster">
            {poster_html}
        </div>
        <div class="anime-card-content">
            <div class="anime-title">{_escape(title or "Unknown Anime")}</div>
            {english_html}
            <div class="anime-meta">{meta_html}</div>
            {genres_html}
            {themes_html}
            {studios_html}
        </div>
    </div>
    """


def build_retrieved_cards(results: list[dict[str, Any]]) -> str:
    """Membuat kumpulan kartu dari hasil retrieval."""
    if not results:
        return ""

    cards = []
    for result in results:
        if not isinstance(result, dict):
            continue
        cards.append(build_anime_card(result))

    if not cards:
        return ""

    return f'<div class="anime-results">{"".join(cards)}</div>'


def build_interleaved_response(answer: str, results: list[dict[str, Any]]) -> str:
    """
    Menggabungkan jawaban SLM dengan kartu anime.
    Jawaban SLM tetap ditampilkan sebagai teks.
    Poster dan metadata ditambahkan dari hasil retrieval, bukan dari output LLM.
    """
    answer = _valid_text(answer)
    cards_html = build_retrieved_cards(results)

    if not cards_html:
        return answer

    if answer:
        return f"""
        <div class="anirag-answer">{answer}</div>
        {cards_html}
        """

    return cards_html


# ============================================================
# HISTORY NORMALIZATION (AniRAG-v2)
# ============================================================

def normalize_history(history: Any) -> list[dict[str, Any]]:
    """
    Menormalisasi history Gradio agar sesuai dengan format
    history yang diharapkan RagPipeline.
    """
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
# CHAT RESPONSE CALLBACK (Logika Backend AniRAG-v2)
# ============================================================

def respond(message: str, history: list[dict[str, Any]] | None):
    """
    Main UI callback.
    Flow: RagPipeline.generate() -> Guardrail -> Multi-turn -> Retrieval -> SLM -> Response
    """
    message = _valid_text(message)

    if not message:
        return history or []

    normalized_history = normalize_history(history)

    try:
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
        print("=" * 70)
        print(repr(exc))
        print("=" * 70 + "\n")

        error_message = "Maaf, terjadi kesalahan saat memproses permintaan. Silakan coba lagi."
        new_history = list(history or [])
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": error_message})
        return new_history

    # BLOCKED / REFUSAL
    if result.get("blocked"):
        refusal = _valid_text(result.get("refusal") or result.get("response"))
        if not refusal:
            refusal = "Maaf, saya hanya dapat membantu pertanyaan yang berkaitan dengan anime."

        new_history = list(history or [])
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": refusal})
        return new_history

    # NORMAL RESPONSE
    answer = _valid_text(result.get("response"))
    results = result.get("results", [])
    if not isinstance(results, list):
        results = []

    formatted_response = build_interleaved_response(answer, results)

    new_history = list(history or [])
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": formatted_response})

    return new_history


# ============================================================
# TAMPILAN HTML, CSS, & JS (100% Mengikuti Project Acuan)
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
    --accent: #111114;       /* bubble user & elemen aksi utama -- hitam netral */
    --accent-soft: #F0F0F2;  /* latar tombol sekunder (mis. tombol kirim) */
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-page) !important;
    font-family: 'Inter', -apple-system, "Segoe UI", Roboto, sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 16px !important;
}

/* ---------- Full-screen shell ---------- */
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

/* ---------- Header: strip putih flat, full-width ---------- */
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

/* ---------- Area chat ---------- */
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

/* Empty-state */
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

/* ---------- Bubble chat ---------- */
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

/* ---------- Styling Kartu Anime AniRAG-v2 dalam Chat ---------- */
.anirag-answer {
    line-height: 1.75;
    margin-bottom: 14px;
}
.anime-results {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 12px;
    margin-top: 12px;
}
.anime-card {
    display: flex;
    gap: 12px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--bg-surface);
}
.anime-card-poster {
    flex: 0 0 86px;
    width: 86px;
}
.anime-poster {
    display: block;
    width: 86px;
    height: 124px;
    object-fit: cover;
    border-radius: 8px;
    background: var(--bg-page);
}
.anime-poster-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 11px;
    border-radius: 8px;
    background: var(--bg-page);
}
.anime-card-content {
    flex: 1;
    min-width: 0;
}
.anime-title {
    font-size: 14.5px;
    font-weight: 700;
    color: var(--text-primary);
}
.anime-english-title {
    color: var(--text-secondary);
    font-size: 11.5px;
    margin-top: 2px;
}
.anime-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
}
.anime-meta-item, .anime-score {
    display: inline-flex;
    align-items: center;
    padding: 2px 6px;
    border-radius: 999px;
    background: var(--bg-page);
    color: var(--text-secondary);
    font-size: 10px;
}
.anime-score {
    font-weight: 600;
}
.anime-detail {
    display: flex;
    flex-direction: column;
    margin-top: 6px;
    font-size: 11px;
    color: var(--text-secondary);
}
.anime-label {
    color: var(--text-muted);
    font-size: 9.5px;
    font-weight: 600;
}

/* ---------- Bottom dock: contoh prompt + kolom input + disclaimer ---------- */
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

/* ---------- MOBILE FIX (<=640px) ---------- */
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
    .anime-results { grid-template-columns: 1fr; }
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
# GRADIO APP BUILD (Struktur UI Referensi)
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

    # Handler tombol contoh
    for btn, prompt_text in zip(example_buttons, EXAMPLE_PROMPTS):
        btn.click(fn=lambda p=prompt_text: p, inputs=None, outputs=msg)

    # Event Submit & Klik Kirim (Menjalankan respond AniRAG-v2)
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
    demo.launch(theme=theme, css=CUSTOM_CSS, js=CLEANUP_JS, share=True)