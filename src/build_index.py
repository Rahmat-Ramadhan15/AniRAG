"""
AniRAG-v2
Phase 4: Embedding + FAISS Indexing

Pipeline:
    anime_documents.jsonl
        ↓
    SentenceTransformer
        ↓
    normalized embeddings
        ↓
    FAISS IndexFlatIP
        ↓
    anime.index + id_mapping.pkl

Input:
    data/processed/anime_documents.jsonl

Output:
    data/index/anime.index
    data/index/id_mapping.pkl

Baseline:
    15,966 documents
    all-MiniLM-L6-v2
    384-dimensional embeddings
    L2-normalized
    FAISS IndexFlatIP

Device:
    auto-detect CPU/GPU
"""

import json
import pickle
from pathlib import Path

import faiss
import yaml
from sentence_transformers import SentenceTransformer


# Configuration

CONFIG_PATH = "configs/config.yaml"


# Configuration Loader

def load_config(path: str = CONFIG_PATH) -> dict:
    """Load centralized project configuration."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config tidak ditemukan: {config_path}"
        )

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Format config.yaml tidak valid.")

    return config


# Document Loader

def load_documents(path: str):
    """
    Load mal_id dan text langsung dari JSONL.

    Urutan mal_id dipertahankan agar:
        vector[i] <-> mal_id[i]

    Returns:
        mal_ids: list[int]
        texts: list[str]
    """
    document_path = Path(path)

    if not document_path.exists():
        raise FileNotFoundError(
            f"File documents tidak ditemukan: {document_path}"
        )

    mal_ids = []
    texts = []

    with document_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL tidak valid pada baris {line_number}: {exc}"
                ) from exc

            if "mal_id" not in document:
                raise ValueError(
                    f"'mal_id' tidak ditemukan pada baris {line_number}"
                )

            if "text" not in document:
                raise ValueError(
                    f"'text' tidak ditemukan pada baris {line_number}"
                )

            mal_id = document["mal_id"]
            text = document["text"]

            if mal_id is None:
                raise ValueError(
                    f"mal_id kosong pada baris {line_number}"
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"text kosong pada baris {line_number}"
                )

            mal_ids.append(mal_id)
            texts.append(text)

    return mal_ids, texts


# Device Detection

def detect_device(configured_device: str = "auto") -> str:
    """
    Detect embedding device.

    Supported:
        auto
        cpu
        cuda
    """
    configured_device = configured_device.lower().strip()

    if configured_device == "cpu":
        return "cpu"

    if configured_device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Device 'cuda' dipilih tetapi CUDA tidak tersedia."
                )

            return "cuda"

        except ImportError as exc:
            raise RuntimeError(
                "PyTorch diperlukan untuk menggunakan device 'cuda'."
            ) from exc

    if configured_device != "auto":
        raise ValueError(
            "embedding.device harus 'auto', 'cpu', atau 'cuda'."
        )

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    except ImportError:
        return "cpu"


# Validation

def validate_documents(
    mal_ids,
    texts,
    expected_documents: int,
    expected_unique_mal_id: int,
):
    """Validate document corpus before expensive embedding."""

    print("[INFO] Validasi corpus...")

    # Number of documents

    document_count = len(texts)

    if document_count != expected_documents:
        raise ValueError(
            f"Jumlah dokumen tidak sesuai: "
            f"{document_count} != {expected_documents}"
        )

    print(f"[OK] Jumlah dokumen = {document_count}")

    # mal_id count

    if len(mal_ids) != document_count:
        raise ValueError(
            "Jumlah mal_id tidak sama dengan jumlah dokumen."
        )

    print(f"[OK] Jumlah mal_id = {len(mal_ids)}")

    # Unique mal_id

    unique_mal_ids = len(set(mal_ids))

    if unique_mal_ids != expected_unique_mal_id:
        raise ValueError(
            f"Jumlah unique mal_id tidak sesuai: "
            f"{unique_mal_ids} != {expected_unique_mal_id}"
        )

    print(f"[OK] Unique mal_id = {unique_mal_ids}")

    # Empty text

    empty_text_count = sum(
        1 for text in texts
        if not isinstance(text, str) or not text.strip()
    )

    if empty_text_count > 0:
        raise ValueError(
            f"Ditemukan {empty_text_count} dokumen dengan text kosong."
        )

    print("[OK] Tidak ada text kosong")


# Build FAISS Index

def build_index(
    documents_path: str,
    model_name: str,
    out_dir: str,
    expected_documents: int,
    expected_unique_mal_id: int,
    expected_dimension: int,
    device: str = "cpu",
    batch_size: int = 64,
    normalize_embeddings: bool = True,
):
    """
    Build embeddings and FAISS index from anime documents.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load documents

    print("=" * 60)
    print("PHASE 4 — EMBEDDING + FAISS INDEXING")
    print("=" * 60)

    print(f"[INFO] Input: {documents_path}")

    mal_ids, texts = load_documents(documents_path)

    print(
        f"[INFO] Dokumen dimuat: {len(texts)} "
        f"(device={device})"
    )

    # Validate corpus

    validate_documents(
        mal_ids=mal_ids,
        texts=texts,
        expected_documents=expected_documents,
        expected_unique_mal_id=expected_unique_mal_id,
    )

    # Load embedding model

    print()
    print("[INFO] Memuat embedding model...")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Device: {device}")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    # Encode documents

    print()
    print("[INFO] Membentuk embeddings...")
    print(f"[INFO] Batch size: {batch_size}")
    print(
        f"[INFO] Normalize embeddings: "
        f"{normalize_embeddings}"
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32")

    # Validate embedding shape

    if embeddings.ndim != 2:
        raise ValueError(
            f"Format embedding tidak valid: "
            f"ndim={embeddings.ndim}"
        )

    embedding_count, dimension = embeddings.shape

    print()
    print(
        f"[INFO] Embedding shape: "
        f"{embedding_count} × {dimension}"
    )

    if embedding_count != expected_documents:
        raise ValueError(
            f"Jumlah embedding tidak sesuai: "
            f"{embedding_count} != {expected_documents}"
        )

    if dimension != expected_dimension:
        raise ValueError(
            f"Dimensi embedding tidak sesuai: "
            f"{dimension} != {expected_dimension}"
        )

    print(
        f"[OK] Jumlah embedding = {embedding_count}"
    )

    print(
        f"[OK] Dimensi embedding = {dimension}"
    )

    # Validate normalization

    if normalize_embeddings:
        import numpy as np

        norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                "Embedding tidak berhasil dinormalisasi."
            )

        print("[OK] Embeddings ter-normalisasi L2")

    # Build FAISS index

    print()
    print("[INFO] Membuat FAISS IndexFlatIP...")

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Validate FAISS index

    if index.ntotal != expected_documents:
        raise ValueError(
            f"Jumlah vector FAISS tidak sesuai: "
            f"{index.ntotal} != {expected_documents}"
        )

    if index.d != expected_dimension:
        raise ValueError(
            f"Dimensi FAISS tidak sesuai: "
            f"{index.d} != {expected_dimension}"
        )

    if len(mal_ids) != index.ntotal:
        raise ValueError(
            f"Mismatch FAISS dan mapping: "
            f"{index.ntotal} vectors vs "
            f"{len(mal_ids)} mal_ids"
        )

    print(f"[OK] FAISS vectors = {index.ntotal}")
    print(f"[OK] FAISS dimension = {index.d}")
    print("[OK] FAISS ↔ mal_id mapping konsisten")

    # Output paths

    index_path = out_dir / "anime.index"
    mapping_path = out_dir / "id_mapping.pkl"

    # Save FAISS index

    faiss.write_index(
        index,
        str(index_path),
    )

    # Save ID mapping

    with mapping_path.open("wb") as f:
        pickle.dump(
            mal_ids,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Final validation

    if not index_path.exists():
        raise RuntimeError(
            f"FAISS index gagal dibuat: {index_path}"
        )

    if not mapping_path.exists():
        raise RuntimeError(
            f"ID mapping gagal dibuat: {mapping_path}"
        )

    print()
    print("=" * 60)
    print("PHASE 4 SELESAI")
    print("=" * 60)
    print(f"Dataset/documents : {expected_documents}")
    print(f"Embeddings        : {embedding_count}")
    print(f"Dimension         : {dimension}")
    print(f"FAISS vectors     : {index.ntotal}")
    print(f"FAISS metric      : Inner Product")
    print(f"Normalization     : {normalize_embeddings}")
    print(f"Model             : {model_name}")
    print(f"Device             : {device}")
    print(f"Index             : {index_path}")
    print(f"Mapping           : {mapping_path}")
    print("=" * 60)

    return index, mal_ids


# Main

if __name__ == "__main__":

    cfg = load_config()

    # Read configuration

    documents_path = cfg["data"]["documents_path"]
    out_dir = cfg["data"]["index_dir"]

    expected_documents = cfg["corpus"]["expected_documents"]
    expected_unique_mal_id = cfg["corpus"]["expected_unique_mal_id"]

    model_name = cfg["embedding"]["model_name"]
    expected_dimension = cfg["embedding"]["expected_dimension"]

    configured_device = cfg["embedding"].get(
        "device",
        "auto",
    )

    batch_size = cfg["embedding"].get(
        "batch_size",
        64,
    )

    normalize_embeddings = cfg["embedding"].get(
        "normalize_embeddings",
        True,
    )

    # Validate embedding configuration

    if model_name != "sentence-transformers/all-MiniLM-L6-v2":
        raise ValueError(
            "Model embedding tidak sesuai baseline AniRAG-v2: "
            f"{model_name}"
        )

    if expected_dimension != 384:
        raise ValueError(
            "Expected embedding dimension AniRAG-v2 harus 384."
        )

    if not normalize_embeddings:
        raise ValueError(
            "normalize_embeddings harus True "
            "untuk FAISS IndexFlatIP berbasis cosine similarity."
        )

    # Detect device

    device = detect_device(
        configured_device
    )

    print(f"[INFO] Device terdeteksi: {device}")

    # Build index

    build_index(
        documents_path=documents_path,
        model_name=model_name,
        out_dir=out_dir,
        expected_documents=expected_documents,
        expected_unique_mal_id=expected_unique_mal_id,
        expected_dimension=expected_dimension,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )