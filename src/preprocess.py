from pathlib import Path
import json
import pandas as pd


# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_anime_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "anime_documents.jsonl"

EXPECTED_COLUMNS = [
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

EXPECTED_ROWS = 15966

BLOCKED_TAGS = {
    "hentai",
    "erotica",
}


# HELPER FUNCTIONS

def normalize_text(value):
    """
    Mengubah nilai menjadi string yang bersih.
    Nilai NaN dikembalikan sebagai string kosong.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()

def format_number(value):
    """
    Format nilai numerik untuk teks embedding.

    pandas membaca kolom numerik dengan NaN sebagai float64, sehingga nilai
    bulat seperti episodes/year ikut tampil dengan akhiran ".0" (mis. "26.0",
    "1998.0") kalau hanya di-str()/normalize_text() langsung. Fungsi ini
    menghilangkan ".0" HANYA kalau nilainya benar-benar bilangan bulat --
    nilai desimal asli (mis. score 8.75) tetap ditampilkan apa adanya.
    """
    if pd.isna(value):
        return ""

    try:
        f = float(value)
    except (TypeError, ValueError):
        return normalize_text(value)

    if f.is_integer():
        return str(int(f))

    return str(f)

def contains_blocked_tag(value):
    """
    Memeriksa apakah sebuah field mengandung tag terlarang.
    Digunakan hanya sebagai validasi, bukan filtering ulang.

    Dataset menggunakan format pipe-delimited (mis. "Action | Hentai"),
    BUKAN comma-delimited -- delimiter diperbaiki dari "," menjadi "|"
    (lihat temuan audit Phase 3: versi sebelumnya tidak pernah mendeteksi
    tag terlarang yang bergabung dengan genre/tema lain dalam satu field).
    """
    text = normalize_text(value).lower()

    if not text:
        return False

    tags = {
        tag.strip().lower()
        for tag in text.split("|")
        if tag.strip()
    }

    return bool(tags.intersection(BLOCKED_TAGS))


def format_metadata_list(value):
    """
    Membersihkan field seperti genres, themes, dan studios.

    Delimiter sumber data adalah "|" (mis. "Action | Award Winning | Sci-Fi"),
    BUKAN ",". Output dipertahankan memakai format " | " yang sama supaya
    konsisten dengan representasi asli data (lihat temuan audit Phase 3).
    """
    value = normalize_text(value)

    if not value:
        return ""

    return " | ".join(
        item.strip()
        for item in value.split("|")
        if item.strip()
    )


# LOAD DATASET

def load_dataset():
    print("=" * 60)
    print("PHASE 3 — BUILD ANIME DOCUMENTS")
    print("=" * 60)

    print(f"[INFO] Input : {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"[OK] Dataset dimuat")
    print(f"[INFO] Shape : {df.shape}")

    return df


# VALIDATE DATASET

def validate_dataset(df):
    print("\n[INFO] Validasi dataset final...")

    # Row count

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Jumlah baris tidak sesuai. "
            f"Expected={EXPECTED_ROWS}, Actual={len(df)}"
        )

    print(f"[OK] Jumlah baris = {len(df)}")

    # Column validation

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        missing = [
            col for col in EXPECTED_COLUMNS
            if col not in actual_columns
        ]

        extra = [
            col for col in actual_columns
            if col not in EXPECTED_COLUMNS
        ]

        raise ValueError(
            "\nSchema dataset tidak sesuai.\n"
            f"Missing columns : {missing}\n"
            f"Extra columns   : {extra}\n"
            f"Actual columns  : {actual_columns}"
        )

    print("[OK] Schema 12 kolom sesuai")

    # MAL ID

    if df["mal_id"].isna().any():
        raise ValueError(
            "Terdapat mal_id yang kosong."
        )

    duplicate_ids = df["mal_id"].duplicated().sum()

    if duplicate_ids > 0:
        raise ValueError(
            f"Terdapat {duplicate_ids} duplicate mal_id."
        )

    print("[OK] mal_id valid dan unik")

    # Blocked tags validation

    blocked_rows = []

    for idx, row in df.iterrows():

        genres = normalize_text(row["genres"])
        themes = normalize_text(row["themes"])

        if (
            contains_blocked_tag(genres)
            or contains_blocked_tag(themes)
        ):
            blocked_rows.append({
                "row": idx,
                "mal_id": row["mal_id"],
                "title": row["title"],
                "genres": genres,
                "themes": themes,
            })

    if blocked_rows:

        print("\n[WARNING] Masih ditemukan blocked tags:")

        for item in blocked_rows[:10]:
            print(
                f"  - MAL ID {item['mal_id']} | "
                f"{item['title']} | "
                f"genres={item['genres']} | "
                f"themes={item['themes']}"
            )

        raise ValueError(
            f"\nDataset belum aman untuk indexing. "
            f"Ditemukan {len(blocked_rows)} baris dengan "
            f"blocked tags."
        )

    print("[OK] Tidak ada Hentai/Erotica pada genres/themes")

    # Core fields

    required_fields = [
        "mal_id",
        "title",
        "genres",
        "synopsis",
    ]

    for column in required_fields:

        missing_count = df[column].isna().sum()

        if missing_count > 0:
            raise ValueError(
                f"Kolom '{column}' memiliki "
                f"{missing_count} nilai kosong."
            )

    print("[OK] Core fields tidak memiliki missing value")

    return True


# BUILD DOCUMENT

def build_document(row):
    """
    Membentuk satu dokumen teks terstruktur untuk embedding.

    image_url TIDAK dimasukkan ke teks embedding karena
    merupakan metadata untuk kebutuhan poster/UI.
    """

    title = normalize_text(row["title"])
    title_english = normalize_text(row["title_english"])

    anime_type = normalize_text(row["type"])
    episodes = format_number(row["episodes"])
    score = format_number(row["score"])
    year = format_number(row["year"])

    genres = format_metadata_list(row["genres"])
    themes = format_metadata_list(row["themes"])
    studios = format_metadata_list(row["studios"])

    synopsis = normalize_text(row["synopsis"])

    document_parts = [
        f"Title: {title}",
    ]

    if title_english:
        document_parts.append(
            f"English Title: {title_english}"
        )

    if anime_type:
        document_parts.append(
            f"Type: {anime_type}"
        )

    if episodes:
        document_parts.append(
            f"Episodes: {episodes}"
        )

    if score:
        document_parts.append(
            f"Score: {score}"
        )

    if year:
        document_parts.append(
            f"Year: {year}"
        )

    if genres:
        document_parts.append(
            f"Genres: {genres}"
        )

    if themes:
        document_parts.append(
            f"Themes: {themes}"
        )

    if studios:
        document_parts.append(
            f"Studios: {studios}"
        )

    if synopsis:
        document_parts.append(
            f"Synopsis: {synopsis}"
        )

    return "\n".join(document_parts)


# BUILD DOCUMENT CORPUS

def build_documents(df):

    print("\n[INFO] Membentuk document corpus...")

    documents = []

    for _, row in df.iterrows():

        document = {
            "mal_id": int(row["mal_id"]),
            "text": build_document(row),
            "metadata": {
                "title": normalize_text(row["title"]),
                "title_english": normalize_text(
                    row["title_english"]
                ),
                "type": normalize_text(row["type"]),
                "episodes": (
                    int(row["episodes"])
                    if pd.notna(row["episodes"])
                    else None
                ),
                "score": (
                    float(row["score"])
                    if pd.notna(row["score"])
                    else None
                ),
                "genres": format_metadata_list(
                    row["genres"]
                ),
                "synopsis": normalize_text(
                    row["synopsis"]
                ),
                "year": (
                    int(row["year"])
                    if pd.notna(row["year"])
                    else None
                ),
                "themes": format_metadata_list(
                    row["themes"]
                ),
                "studios": format_metadata_list(
                    row["studios"]
                ),
                "image_url": normalize_text(
                    row["image_url"]
                ),
            },
        }

        documents.append(document)

    return documents


# SAVE JSONL

def save_documents(documents):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\n[INFO] Menyimpan documents...")
    print(f"[INFO] Output: {OUTPUT_FILE}")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for document in documents:

            f.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"[OK] {len(documents)} documents berhasil disimpan"
    )


# VALIDATE OUTPUT

def validate_output(documents):

    print("\n[INFO] Validasi output...")

    if len(documents) != EXPECTED_ROWS:
        raise ValueError(
            f"Jumlah document tidak sesuai. "
            f"Expected={EXPECTED_ROWS}, "
            f"Actual={len(documents)}"
        )

    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            "File output tidak ditemukan."
        )

    # Validasi setiap document
    required_document_keys = {
        "mal_id",
        "text",
        "metadata",
    }

    for i, document in enumerate(documents):

        if not required_document_keys.issubset(
            document.keys()
        ):
            raise ValueError(
                f"Document ke-{i} memiliki struktur tidak valid."
            )

        if not document["text"].strip():
            raise ValueError(
                f"Document ke-{i} memiliki text kosong."
            )

    print("[OK] Jumlah document sesuai")
    print("[OK] Struktur document valid")
    print("[OK] Tidak ada document kosong")

    print("\n" + "=" * 60)
    print("PHASE 3 SELESAI")
    print("=" * 60)

    print(f"Dataset      : {len(documents)} rows")
    print(f"Documents    : {len(documents)}")
    print(f"Output       : {OUTPUT_FILE}")
    print("=" * 60)

# MAIN

def main():

    df = load_dataset()

    validate_dataset(df)

    documents = build_documents(df)

    save_documents(documents)

    validate_output(documents)


if __name__ == "__main__":
    main()