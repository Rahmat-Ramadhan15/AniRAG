# ==============================================================
# AniRAG-v2
# RAG PIPELINE — UNIFIED
#
# Phase 5A : Artifact loading + consistency validation
# Phase 5B : Semantic retrieval using FAISS
# Phase 5C : Attribute filtering + anchor/franchise handling
#            + reranking + anchor-based semantic retrieval
# Phase 6  : Guardrail hook
# Phase 7  : Multi-turn refinement (2 turns)
# SLM      : Llama-3.2-3B-Instruct
#
# PRINCIPLE:
# - Phase 5C READY TO FREEZE remains the retrieval baseline.
# - SLM, guardrail, and multi-turn are integrated around it.
# - No rebuild of dataset / embedding / FAISS is required.
# ==============================================================

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================
# OPTIONAL GUARDRAIL IMPORT
# ==============================================================

try:
    from .guardrails import guard_query
except ImportError:
    try:
        from guardrails import guard_query
    except ImportError:
        # Fallback only for compatibility.
        # In the actual project, guardrails.py should exist.
        def guard_query(query: str):
            return None


# ==============================================================
# MULTI-TURN
# ==============================================================

FOLLOWUP_HINTS = [
    # Constraint / refinement yang spesifik
    "yang rating",
    "yang skornya",
    "dengan score",
    "dengan skor",
    "di atas",
    "di bawah",
    "lebih dari",
    "kurang dari",
    "yang lebih",
    "yang kurang",

    # Rujukan terhadap percakapan sebelumnya
    "sebelumnya",
    "tadi",
    "barusan",
    "yang lain",
    "lainnya",
    "itu tadi",
    "seperti itu",
    "yang serupa",
    "mirip itu",
]


# ==============================================================
# SYSTEM PROMPTS
# ==============================================================

SYSTEM_PROMPT_RAG = """
Anda adalah AniRAG, chatbot rekomendasi anime berbasis RAG dan SLM.

ATURAN UTAMA:
1. Gunakan HANYA informasi anime yang tersedia pada CONTEXT.
2. Jangan mengarang fakta yang tidak ada di CONTEXT.
3. ATURAN JUMLAH REKOMENDASI:
   - Jika pengguna TIDAK menyebutkan jumlah spesifik (misal: "rekomendasikan anime romance", "ada yang mirip Naruto?"), berikan HANYA 1 REKOMENDASI TERBAIK.
   - Jika pengguna menyebutkan jumlah spesifik (misal: "rekomendasikan 3 anime"), berikan sesuai jumlah yang diminta.

4. FORMAT JAWABAN (SANGAT WAJIB):
   Setiap anime yang direkomendasikan WAJIB ditulis persis menggunakan format berikut:

### [Nama Judul Anime Eksak Sesuai Context]
Plot: [Penjelasan ringkas plot/jalan cerita dari anime]
Alasan: [Alasan mengapa anime ini cocok dengan permintaan pengguna]

---

Contoh:
### Naruto Shippuden
Plot: Naruto Uzumaki kembali ke desa Konoha setelah berlatih selama dua setengah tahun untuk menghadapi ancaman organisasi Akatsuki.
Alasan: Anime ini sangat cocok jika Anda menyukai aksi ninja dengan pertarungan yang intens dan jalan cerita yang penuh emosi.

5. Jika pertanyaan berupa informasi faktual singkat (seperti "berapa episode FMA Brotherhood?"):
   Jawab langsung secara rinci dan singkat tanpa perlu menggunakan format '###'.

CONTEXT:
{context}
"""


SYSTEM_PROMPT_BASELINE = """
Anda adalah chatbot rekomendasi anime AniRAG.

Anda sedang digunakan sebagai BASELINE SLM-ONLY.

Jangan mengklaim menggunakan database AniRAG, FAISS,
retrieval, atau context eksternal.

Jawablah pertanyaan secara natural dan ringkas.
Jika tidak mengetahui suatu informasi, katakan bahwa
Anda tidak yakin daripada mengarang fakta.
"""


PROMPT_TEMPLATE = """
Pertanyaan pengguna:
{query}

Berikan jawaban dalam Bahasa Indonesia.
"""


# ==============================================================
# RAG PIPELINE
# ==============================================================

class RagPipeline:
    """
    Unified RAG pipeline AniRAG-v2.

    Phase 5A
        - Load config
        - Load documents
        - Load FAISS index
        - Load ID mapping
        - Validate artifact consistency

    Phase 5B
        - Load embedding model
        - Encode query
        - Semantic retrieval

    Phase 5C
        - Attribute filtering
        - Anchor detection
        - Anchor-based semantic retrieval
        - Franchise exclusion
        - Reranking

    Phase 6
        - Guardrail hook

    Phase 7
        - 2-turn follow-up refinement

    SLM
        - Llama-3.2-3B-Instruct
    """

    # ==========================================================
    # TAXONOMY — SOURCE DATA
    # ==========================================================

    KNOWN_GENRES = {
        "action",
        "adventure",
        "avant garde",
        "award winning",
        "boys love",
        "comedy",
        "drama",
        "fantasy",
        "girls love",
        "gourmet",
        "horror",
        "mystery",
        "romance",
        "sci-fi",
        "slice of life",
        "sports",
        "supernatural",
        "suspense",
        "ecchi",
    }

    KNOWN_THEMES = {
        "adult cast",
        "anthropomorphic",
        "cgdct",
        "childcare",
        "combat sports",
        "crossdressing",
        "delinquents",
        "detective",
        "educational",
        "gag humor",
        "gore",
        "harem",
        "high stakes game",
        "historical",
        "idols (female)",
        "idols (male)",
        "isekai",
        "iyashikei",
        "love polygon",
        "love status quo",
        "magical sex shift",
        "mahou shoujo",
        "martial arts",
        "mecha",
        "medical",
        "military",
        "music",
        "mythology",
        "organized crime",
        "otaku culture",
        "parody",
        "performing arts",
        "pets",
        "psychological",
        "racing",
        "reincarnation",
        "reverse harem",
        "samurai",
        "school",
        "showbiz",
        "space",
        "strategy game",
        "super power",
        "survival",
        "team sports",
        "time travel",
        "urban fantasy",
        "vampire",
        "video game",
        "villainess",
        "visual arts",
        "workplace",
    }

    KNOWN_TYPES = {
        "tv",
        "movie",
        "ova",
        "ona",
    }

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
    ):

        self.config_path = PROJECT_ROOT / config_path

        # ------------------------------------------------------
        # Config
        # ------------------------------------------------------

        self.config = self._load_config()

        # ------------------------------------------------------
        # Paths
        # ------------------------------------------------------

        self.documents_path = (
            PROJECT_ROOT
            / self.config["data"]["documents_path"]
        )

        self.index_dir = (
            PROJECT_ROOT
            / self.config["data"]["index_dir"]
        )

        self.index_path = (
            self.index_dir / "anime.index"
        )

        self.mapping_path = (
            self.index_dir / "id_mapping.pkl"
        )

        # ------------------------------------------------------
        # Embedding configuration
        # ------------------------------------------------------

        self.embedding_model_name = (
            self.config["embedding"]["model_name"]
        )

        configured_device = (
            self.config["embedding"]["device"]
        )

        self.embedding_device = self._detect_device(
            configured_device
        )

        self.embedding_dimension = int(
            self.config["embedding"]["expected_dimension"]
        )

        self.normalize_embeddings = bool(
            self.config["embedding"]["normalize_embeddings"]
        )

        # ------------------------------------------------------
        # Retrieval configuration
        # ------------------------------------------------------

        self.retrieval_top_k = int(
            self.config["retrieval"]["top_k_final"]
        )

        configured_candidates = (
            self.config["retrieval"].get(
                "top_k_candidates",
                [self.retrieval_top_k],
            )
        )

        if not isinstance(
            configured_candidates,
            list,
        ):
            raise TypeError(
                "retrieval.top_k_candidates "
                "harus berupa list."
            )

        self.retrieval_candidate_k = max(
            [
                int(k)
                for k in configured_candidates
            ]
            + [self.retrieval_top_k]
        )

        # Minimum pool untuk query dengan
        # hard filter atau anchor.
        self.hard_filter_candidate_k = 50

        # ------------------------------------------------------
        # Reranking configuration
        # ------------------------------------------------------

        self.semantic_weight = 0.6
        self.score_weight = 0.4

        if abs(
            self.semantic_weight
            + self.score_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "Bobot reranking harus berjumlah 1.0."
            )

        # ------------------------------------------------------
        # LLM state
        # ------------------------------------------------------

        self.llm = None
        self.tokenizer = None
        self.text_generation_pipeline = None

        # ------------------------------------------------------
        # Validate artifacts
        # ------------------------------------------------------

        self._validate_artifact_paths()

        # ------------------------------------------------------
        # Load documents
        # ------------------------------------------------------

        self.documents = self._load_documents()

        # ------------------------------------------------------
        # Build document lookup
        # ------------------------------------------------------

        self.document_lookup = (
            self._build_document_lookup()
        )

        # ------------------------------------------------------
        # Static metadata caches
        # ------------------------------------------------------

        self.known_studios = (
            self._build_known_studios()
        )

        self.title_anchors = (
            self._build_title_anchors()
        )

        print(
            f"[INFO] Known studios cached: "
            f"{len(self.known_studios)}"
        )

        print(
            f"[INFO] Title anchors cached: "
            f"{len(self.title_anchors)}"
        )

        # ------------------------------------------------------
        # FAISS
        # ------------------------------------------------------

        self.index = self._load_faiss_index()

        # ------------------------------------------------------
        # ID mapping
        # ------------------------------------------------------

        self.id_mapping = (
            self._load_id_mapping()
        )

        # ------------------------------------------------------
        # Consistency
        # ------------------------------------------------------

        self._validate_consistency()

        # ------------------------------------------------------
        # Embedding model
        # ------------------------------------------------------

        print(
            f"[INFO] Memuat embedding model: "
            f"{self.embedding_model_name}"
        )

        print(
            f"[INFO] Embedding device: "
            f"{self.embedding_device}"
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.embedding_device,
        )

        model_dimension = (
            self.embedding_model
            .get_embedding_dimension()
        )

        if model_dimension != self.embedding_dimension:
            raise ValueError(
                "Dimensi embedding model tidak sesuai. "
                f"Expected={self.embedding_dimension}, "
                f"Actual={model_dimension}"
            )

        print(
            f"[OK] Embedding dimension = "
            f"{model_dimension}"
        )

        print(
            "[OK] Embedding model berhasil dimuat"
        )

    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    def _load_config(
        self,
    ) -> Dict[str, Any]:

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config tidak ditemukan: "
                f"{self.config_path}"
            )

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(
                "config.yaml harus berupa dictionary/object."
            )

        required_sections = [
            "data",
            "corpus",
            "embedding",
            "retrieval",
            "llm",
        ]

        missing_sections = [
            section
            for section in required_sections
            if section not in config
        ]

        if missing_sections:
            raise KeyError(
                "Section config tidak ditemukan: "
                + ", ".join(missing_sections)
            )

        return config

    # ==========================================================
    # DEVICE
    # ==========================================================

    def _detect_device(
        self,
        configured_device: str,
    ) -> str:

        import torch

        if not isinstance(
            configured_device,
            str,
        ):
            raise TypeError(
                "embedding.device harus berupa string."
            )

        configured_device = (
            configured_device.strip().lower()
        )

        if configured_device == "auto":

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        elif configured_device == "cpu":

            device = "cpu"

        elif configured_device == "cuda":

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Device 'cuda' diminta secara eksplisit, "
                    "tetapi CUDA tidak tersedia."
                )

            device = "cuda"

        else:

            raise ValueError(
                "Device tidak valid: "
                f"{configured_device}. "
                "Gunakan 'auto', 'cpu', atau 'cuda'."
            )

        print(
            f"[INFO] Device terdeteksi: {device}"
        )

        return device

    # ==========================================================
    # ARTIFACT VALIDATION
    # ==========================================================

    def _validate_artifact_paths(
        self,
    ) -> None:

        required_files = [
            self.documents_path,
            self.index_path,
            self.mapping_path,
        ]

        for path in required_files:

            if not path.exists():

                raise FileNotFoundError(
                    f"Artifact tidak ditemukan: {path}"
                )

    # ==========================================================
    # DOCUMENT LOADING
    # ==========================================================

    def _load_documents(
        self,
    ) -> List[Dict[str, Any]]:

        documents = []

        with self.documents_path.open(
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
                    document = json.loads(line)

                except json.JSONDecodeError as e:

                    raise ValueError(
                        f"JSON tidak valid pada baris "
                        f"{line_number}: {e}"
                    )

                if not isinstance(
                    document,
                    dict,
                ):
                    raise ValueError(
                        f"Document pada baris "
                        f"{line_number} bukan object."
                    )

                required_keys = [
                    "mal_id",
                    "text",
                    "metadata",
                ]

                missing_keys = [
                    key
                    for key in required_keys
                    if key not in document
                ]

                if missing_keys:

                    raise KeyError(
                        f"Document baris {line_number} "
                        f"kehilangan field: {missing_keys}"
                    )

                mal_id = document["mal_id"]
                text = document["text"]
                metadata = document["metadata"]

                if not isinstance(
                    mal_id,
                    int,
                ):
                    raise TypeError(
                        f"mal_id pada baris {line_number} "
                        "harus integer."
                    )

                if not isinstance(
                    text,
                    str,
                ):
                    raise TypeError(
                        f"text pada baris {line_number} "
                        "harus string."
                    )

                if not text.strip():

                    raise ValueError(
                        f"text kosong pada baris {line_number}."
                    )

                if not isinstance(
                    metadata,
                    dict,
                ):
                    raise TypeError(
                        f"metadata pada baris {line_number} "
                        "harus object."
                    )

                documents.append(document)

        print(
            f"[INFO] Dokumen dimuat: {len(documents)}"
        )

        return documents

    # ==========================================================
    # STATIC METADATA CACHE
    # ==========================================================

    def _build_known_studios(
        self,
    ) -> set[str]:

        known_studios = set()

        for document in self.documents:

            metadata = document.get(
                "metadata",
                {},
            )

            studio_values = (
                self._split_metadata_values(
                    metadata.get(
                        "studios",
                        "",
                    )
                )
            )

            known_studios.update(
                studio
                for studio in studio_values
                if studio
            )

        return known_studios

    def _build_title_anchors(
        self,
    ) -> List[Dict[str, Any]]:

        title_anchors = []

        for document in self.documents:

            metadata = document.get(
                "metadata",
                {},
            )

            mal_id = document.get(
                "mal_id"
            )

            for field in (
                "title",
                "title_english",
            ):

                title = metadata.get(field)

                if not title:
                    continue

                normalized_title = (
                    self._normalize_text(
                        str(title)
                    )
                )

                if not normalized_title:
                    continue

                if len(normalized_title) < 3:
                    continue

                title_anchors.append(
                    {
                        "title": str(
                            title
                        ).strip(),
                        "normalized_title": (
                            normalized_title
                        ),
                        "mal_id": mal_id,
                    }
                )

        title_anchors.sort(
            key=lambda item: len(
                item["normalized_title"]
            ),
            reverse=True,
        )

        return title_anchors

    # ==========================================================
    # DOCUMENT LOOKUP
    # ==========================================================

    def _build_document_lookup(
        self,
    ) -> Dict[int, Dict[str, Any]]:

        lookup = {}

        for document in self.documents:

            mal_id = document["mal_id"]

            if mal_id in lookup:

                raise ValueError(
                    f"Duplicate mal_id ditemukan: {mal_id}"
                )

            lookup[mal_id] = document

        return lookup

    # ==========================================================
    # FAISS
    # ==========================================================

    def _load_faiss_index(self):

        print(
            f"[INFO] Memuat FAISS index: "
            f"{self.index_path}"
        )

        index = faiss.read_index(
            str(self.index_path)
        )

        print(
            f"[OK] FAISS index dimuat: "
            f"{index.ntotal} vectors"
        )

        return index

    # Compatibility wrapper.
    def load_index(self):
        return self._load_faiss_index()

    # ==========================================================
    # ID MAPPING
    # ==========================================================

    def _load_id_mapping(
        self,
    ) -> List[int]:

        print(
            f"[INFO] Memuat ID mapping: "
            f"{self.mapping_path}"
        )

        with self.mapping_path.open(
            "rb"
        ) as f:

            mapping = pickle.load(f)

        if not isinstance(
            mapping,
            list,
        ):
            raise TypeError(
                "id_mapping.pkl harus berupa list."
            )

        return mapping

    # ==========================================================
    # CONSISTENCY
    # ==========================================================

    def _validate_consistency(
        self,
    ) -> None:

        expected_documents = int(
            self.config["corpus"][
                "expected_documents"
            ]
        )

        document_ids = [
            document["mal_id"]
            for document in self.documents
        ]

        if len(self.documents) != expected_documents:

            raise ValueError(
                "Jumlah documents tidak sesuai. "
                f"Expected={expected_documents}, "
                f"Actual={len(self.documents)}"
            )

        print(
            f"[OK] Jumlah documents = "
            f"{len(self.documents)}"
        )

        if len(document_ids) != len(
            set(document_ids)
        ):

            raise ValueError(
                "Ditemukan duplicate mal_id pada documents."
            )

        print(
            f"[OK] Unique mal_id = "
            f"{len(document_ids)}"
        )

        if len(self.id_mapping) != len(
            self.documents
        ):

            raise ValueError(
                "Jumlah ID mapping tidak sama "
                "dengan jumlah documents."
            )

        print(
            f"[OK] ID mapping = "
            f"{len(self.id_mapping)}"
        )

        if self.index.ntotal != len(
            self.documents
        ):

            raise ValueError(
                "Jumlah vector FAISS tidak sama "
                "dengan jumlah documents."
            )

        print(
            f"[OK] FAISS vectors = "
            f"{self.index.ntotal}"
        )

        if self.index.d != self.embedding_dimension:

            raise ValueError(
                "Dimensi FAISS tidak sesuai. "
                f"Expected={self.embedding_dimension}, "
                f"Actual={self.index.d}"
            )

        print(
            f"[OK] FAISS dimension = "
            f"{self.index.d}"
        )

        if len(self.id_mapping) != len(
            set(self.id_mapping)
        ):

            raise ValueError(
                "Ditemukan duplicate mal_id pada ID mapping."
            )

        if set(document_ids) != set(
            self.id_mapping
        ):

            raise ValueError(
                "Set mal_id pada documents dan "
                "ID mapping tidak sama."
            )

        for position, mal_id in enumerate(
            self.id_mapping
        ):

            if document_ids[position] != mal_id:

                raise ValueError(
                    "Mapping posisi tidak konsisten "
                    f"pada index {position}: "
                    f"document={document_ids[position]}, "
                    f"mapping={mal_id}"
                )

        print(
            "[OK] FAISS ↔ mal_id ↔ document "
            "mapping konsisten"
        )

    # ==========================================================
    # TEXT NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        value = str(
            value
        ).strip().lower()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    @staticmethod
    def _split_metadata_values(
        value: Any,
    ) -> List[str]:

        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            values = value

        else:
            values = str(value).split("|")

        return [
            str(item).strip().lower()
            for item in values
            if str(item).strip()
        ]

    # ==========================================================
    # QUERY / DOCUMENT EMBEDDING
    # ==========================================================

    def _encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "Query harus berupa string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query tidak boleh kosong."
            )

        embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=(
                self.normalize_embeddings
            ),
            convert_to_numpy=True,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 2:
            raise ValueError(
                "Shape query embedding harus 2D, "
                f"actual={embedding.shape}"
            )

        if embedding.shape != (
            1,
            self.embedding_dimension,
        ):
            raise ValueError(
                "Dimensi query embedding tidak sesuai. "
                f"Expected=(1, {self.embedding_dimension}), "
                f"Actual={embedding.shape}"
            )

        return embedding

    def _encode_anchor_document(
        self,
        anchor: Dict[str, Any],
    ) -> np.ndarray:

        anchor_mal_id = anchor["mal_id"]

        anchor_document = (
            self.document_lookup.get(
                anchor_mal_id
            )
        )

        if anchor_document is None:
            raise KeyError(
                "Dokumen anchor tidak ditemukan: "
                f"mal_id={anchor_mal_id}"
            )

        anchor_text = anchor_document.get(
            "text"
        )

        if not isinstance(
            anchor_text,
            str,
        ) or not anchor_text.strip():

            raise ValueError(
                "Dokumen anchor tidak memiliki text "
                f"yang valid: mal_id={anchor_mal_id}"
            )

        return self._encode_query(
            anchor_text
        )

    # ==========================================================
    # PHASE 5B — RETRIEVAL
    # ==========================================================

    def _search_faiss(
        self,
        embedding: np.ndarray,
        candidate_k: int,
    ) -> List[Dict[str, Any]]:

        similarities, indices = (
            self.index.search(
                embedding,
                candidate_k,
            )
        )

        results = []

        for rank, (
            similarity,
            vector_index,
        ) in enumerate(
            zip(
                similarities[0],
                indices[0],
            ),
            start=1,
        ):

            if vector_index < 0:
                continue

            vector_index = int(
                vector_index
            )

            mal_id = self.id_mapping[
                vector_index
            ]

            document = self.document_lookup.get(
                mal_id
            )

            if document is None:
                raise KeyError(
                    "mal_id dari FAISS mapping "
                    "tidak ditemukan pada documents: "
                    f"{mal_id}"
                )

            results.append(
                {
                    "rank": rank,
                    "mal_id": mal_id,
                    "similarity": float(
                        similarity
                    ),
                    "text": document["text"],
                    "metadata": document["metadata"],
                }
            )

        return results

    def _retrieve_candidates(
        self,
        query: str,
        candidate_k: int,
    ) -> List[Dict[str, Any]]:

        if not isinstance(
            candidate_k,
            int,
        ):
            raise TypeError(
                "candidate_k harus berupa integer."
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k harus lebih besar dari 0."
            )

        candidate_k = min(
            candidate_k,
            self.index.ntotal,
        )

        query_embedding = self._encode_query(
            query
        )

        return self._search_faiss(
            query_embedding,
            candidate_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:

        if top_k is None:
            top_k = self.retrieval_top_k

        return self._retrieve_candidates(
            query,
            top_k,
        )

    # ==========================================================
    # PHASE 5C — ATTRIBUTE DETECTION
    # ==========================================================

    def _detect_attribute_filters(
        self,
        query: str,
    ) -> Dict[str, Any]:

        normalized_query = (
            self._normalize_text(query)
        )

        filters = {
            "genres": [],
            "themes": [],
            "min_score": None,
            "max_score": None,
            "min_year": None,
            "max_year": None,
            "type": None,
            "studios": [],
        }

        # ------------------------------------------------------
        # Genre
        # ------------------------------------------------------

        for genre in sorted(
            self.KNOWN_GENRES,
            key=len,
            reverse=True,
        ):

            if re.search(
                rf"(?<!\w){re.escape(genre)}(?!\w)",
                normalized_query,
            ):

                filters["genres"].append(
                    genre
                )

        # ------------------------------------------------------
        # Theme
        # ------------------------------------------------------

        for theme in sorted(
            self.KNOWN_THEMES,
            key=len,
            reverse=True,
        ):

            if re.search(
                rf"(?<!\w){re.escape(theme)}(?!\w)",
                normalized_query,
            ):

                filters["themes"].append(
                    theme
                )

        # ------------------------------------------------------
        # Score
        # ------------------------------------------------------

        score_min_patterns = [
            (
                r"(?:score|rating|nilai|skor)"
                r"\s*(?:minimal|minimum|setidaknya|"
                r"di atas|diatas|above|>=|>)"
                r"\s*(\d+(?:\.\d+)?)"
            ),
            (
                r"(?:minimal|minimum|setidaknya|"
                r"di atas|diatas|above)"
                r"\s*(?:score|rating|nilai|skor)?"
                r"\s*(\d+(?:\.\d+)?)"
            ),
        ]

        for pattern in score_min_patterns:

            match = re.search(
                pattern,
                normalized_query,
            )

            if match:

                filters["min_score"] = float(
                    match.group(1)
                )

                break

        score_max_patterns = [
            (
                r"(?:score|rating|nilai|skor)"
                r"\s*(?:maksimal|paling tinggi|"
                r"di bawah|dibawah|below|<=|<)"
                r"\s*(\d+(?:\.\d+)?)"
            ),
            (
                r"(?:maksimal|paling tinggi|"
                r"di bawah|dibawah|below)"
                r"\s*(?:score|rating|nilai|skor)?"
                r"\s*(\d+(?:\.\d+)?)"
            ),
        ]

        for pattern in score_max_patterns:

            match = re.search(
                pattern,
                normalized_query,
            )

            if match:

                filters["max_score"] = float(
                    match.group(1)
                )

                break

        # ------------------------------------------------------
        # Year
        # ------------------------------------------------------

        year_min_patterns = [
            (
                r"\btahun\s+"
                r"(\d{4})"
                r"\s+ke\s+atas\b"
            ),
            (
                r"\btahun\s+"
                r"(?:setelah|sesudah|lebih dari)\s+"
                r"(\d{4})\b"
            ),
            (
                r"\b(?:setelah|sesudah|lebih dari)\s+"
                r"(\d{4})\b"
            ),
        ]

        for pattern in year_min_patterns:

            match = re.search(
                pattern,
                normalized_query,
            )

            if match:

                year_value = int(
                    match.group(1)
                )

                # "setelah" / "lebih dari" bersifat strict.
                if re.search(
                    r"(?:setelah|sesudah|lebih dari)",
                    match.group(0),
                ):

                    filters["min_year"] = (
                        year_value + 1
                    )

                else:

                    filters["min_year"] = (
                        year_value
                    )

                break

        year_max_patterns = [
            (
                r"\btahun\s+"
                r"(\d{4})"
                r"\s+ke\s+bawah\b"
            ),
            (
                r"\btahun\s+"
                r"(?:sebelum|kurang dari)\s+"
                r"(\d{4})\b"
            ),
            (
                r"\b(?:sebelum|kurang dari)\s+"
                r"(\d{4})\b"
            ),
        ]

        for pattern in year_max_patterns:

            match = re.search(
                pattern,
                normalized_query,
            )

            if match:

                year_value = int(
                    match.group(1)
                )

                # "sebelum" / "kurang dari" bersifat strict.
                if re.search(
                    r"(?:sebelum|kurang dari)",
                    match.group(0),
                ):

                    filters["max_year"] = (
                        year_value - 1
                    )

                else:

                    filters["max_year"] = (
                        year_value
                    )

                break

        # Exact year hanya digunakan jika
        # tidak ada range year.
        if (
            filters["min_year"] is None
            and filters["max_year"] is None
        ):

            year_match = re.search(
                r"\b(19\d{2}|20\d{2})\b",
                normalized_query,
            )

            if year_match:

                year_value = int(
                    year_match.group(1)
                )

                filters["min_year"] = (
                    year_value
                )

                filters["max_year"] = (
                    year_value
                )

        # ------------------------------------------------------
        # Type
        # ------------------------------------------------------

        for anime_type in sorted(
            self.KNOWN_TYPES,
            key=len,
            reverse=True,
        ):

            if re.search(
                rf"(?<!\w){re.escape(anime_type)}(?!\w)",
                normalized_query,
            ):

                filters["type"] = anime_type

                break

        # ------------------------------------------------------
        # Studio
        # ------------------------------------------------------

        for studio in sorted(
            self.known_studios,
            key=len,
            reverse=True,
        ):

            if re.search(
                rf"(?<!\w){re.escape(studio)}(?!\w)",
                normalized_query,
            ):

                filters["studios"].append(
                    studio
                )

        return filters

    # Compatibility aliases
    def _detect_genre_filter(
        self,
        query: str,
    ) -> List[str]:

        return self._detect_attribute_filters(
            query
        )["genres"]

    def _detect_min_score(
        self,
        query: str,
    ) -> Optional[float]:

        return self._detect_attribute_filters(
            query
        )["min_score"]

    # ==========================================================
    # PHASE 5C — HARD FILTERING
    # ==========================================================

    def _matches_attribute_filters(
        self,
        result: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:

        metadata = result["metadata"]

        # Genre: OR
        required_genres = filters["genres"]

        if required_genres:

            candidate_genres = (
                self._split_metadata_values(
                    metadata.get("genres")
                )
            )

            if not any(
                genre in candidate_genres
                for genre in required_genres
            ):
                return False

        # Theme: OR
        required_themes = filters["themes"]

        if required_themes:

            candidate_themes = (
                self._split_metadata_values(
                    metadata.get("themes")
                )
            )

            if not any(
                theme in candidate_themes
                for theme in required_themes
            ):
                return False

        # Score
        min_score = filters["min_score"]
        max_score = filters["max_score"]

        raw_score = metadata.get("score")

        if min_score is not None or max_score is not None:

            if raw_score is None:
                return False

            try:
                candidate_score = float(
                    raw_score
                )
            except (
                TypeError,
                ValueError,
            ):
                return False

            if (
                min_score is not None
                and candidate_score < min_score
            ):
                return False

            if (
                max_score is not None
                and candidate_score > max_score
            ):
                return False

        # Year
        min_year = filters["min_year"]
        max_year = filters["max_year"]

        if (
            min_year is not None
            or max_year is not None
        ):

            raw_year = metadata.get("year")

            if raw_year is None:
                return False

            try:
                candidate_year = int(
                    float(raw_year)
                )
            except (
                TypeError,
                ValueError,
            ):
                return False

            if (
                min_year is not None
                and candidate_year < min_year
            ):
                return False

            if (
                max_year is not None
                and candidate_year > max_year
            ):
                return False

        # Type
        required_type = filters["type"]

        if required_type is not None:

            candidate_type = (
                self._normalize_text(
                    metadata.get("type")
                )
            )

            if candidate_type != required_type:
                return False

        # Studio
        required_studios = filters["studios"]

        if required_studios:

            candidate_studios = (
                self._split_metadata_values(
                    metadata.get("studios")
                )
            )

            if not any(
                studio in candidate_studios
                for studio in required_studios
            ):
                return False

        return True

    def _apply_attribute_filters(
        self,
        candidates: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        return [
            candidate
            for candidate in candidates
            if self._matches_attribute_filters(
                candidate,
                filters,
            )
        ]

    def _has_hard_filters(
        self,
        filters: Dict[str, Any],
    ) -> bool:

        return any(
            [
                bool(filters["genres"]),
                bool(filters["themes"]),
                filters["min_score"] is not None,
                filters["max_score"] is not None,
                filters["min_year"] is not None,
                filters["max_year"] is not None,
                filters["type"] is not None,
                bool(filters["studios"]),
            ]
        )

    def _determine_candidate_k(
        self,
        filters: Dict[str, Any],
        anchor: Optional[Dict[str, Any]] = None,
    ) -> int:

        if (
            self._has_hard_filters(filters)
            or anchor is not None
        ):

            candidate_k = max(
                self.hard_filter_candidate_k,
                self.retrieval_top_k,
            )

        else:

            candidate_k = max(
                self.retrieval_top_k,
                self.retrieval_candidate_k,
            )

        return min(
            candidate_k,
            self.index.ntotal,
        )

    # ==========================================================
    # FACTUAL ENTITY RESOLUTION
    # ==========================================================

    def _has_factual_intent(self, query: str) -> bool:
        """Deteksi pertanyaan faktual tentang satu anime entity."""
        normalized_query = self._normalize_text(query)

        if not normalized_query:
            return False

        # Query rekomendasi/filter atribut bukan factual entity query.
        recommendation_patterns = [
            r"\brekomendasikan\b",
            r"\brekomendasi\b",
            r"\bcari\b",
            r"\bcarikan\b",
            r"\btampilkan\b",
            r"\bberi(?:kan)?\b",
        ]

        if any(
            re.search(pattern, normalized_query)
            for pattern in recommendation_patterns
        ):
            return False

        factual_patterns = [
            r"\bberapa episode\b",
            r"\bjumlah episode\b",
            r"\bepisode\b",
            r"\bstudio\b",
            r"\bjenis anime\b",
            r"\btipe anime\b",
            r"\btype anime\b",
            r"\btipe\b",
            r"\btype\b",
            r"\bgenre\b",
            r"\btema\b",
            r"\bthemes?\b",
            r"\bscore\b",
            r"\bskor\b",
            r"\bnilai\b",
            r"\btahun tayang\b",
            r"\btahun\b",
            r"\bjudul inggris\b",
            r"\bnama inggris\b",
            r"\benglish title\b",
        ]

        return any(
            re.search(pattern, normalized_query)
            for pattern in factual_patterns
        )

    def _resolve_factual_entity(
        self,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve entity factual hanya dengan full title/title_english."""
        if not self._has_factual_intent(query):
            return None

        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return None

        matches = []

        for anchor in self.title_anchors:
            candidate = anchor.get("normalized_title", "")
            if not candidate:
                continue

            pattern = (
                rf"(?<!\w)"
                f"{re.escape(candidate)}"
                rf"(?!\w)"
            )

            if re.search(pattern, normalized_query):
                matches.append({
                    "mal_id": anchor["mal_id"],
                    "title": anchor["title"],
                    "matched_text": candidate,
                    "match_type": "full",
                })

        if not matches:
            return None

        longest_length = max(
            len(item["matched_text"])
            for item in matches
        )

        longest_matches = [
            item
            for item in matches
            if len(item["matched_text"]) == longest_length
        ]

        unique_mal_ids = {
            item["mal_id"]
            for item in longest_matches
        }

        if len(unique_mal_ids) != 1:
            print(
                "[FACTUAL ENTITY] Ambiguous exact-title match; "
                "resolution aborted."
            )
            return None

        best = longest_matches[0]

        print(
            f"[FACTUAL ENTITY] '{best['matched_text']}' -> "
            f"{best['title']} (mal_id={best['mal_id']})"
        )

        return best

    def _remove_matched_title_span(
        self,
        query: str,
        factual_entity: Dict[str, Any],
    ) -> str:
        """Hapus title yang sudah di-resolve sebelum parsing attribute filter."""
        normalized_query = self._normalize_text(query)
        matched_text = self._normalize_text(
            factual_entity.get("matched_text", "")
        )

        if not normalized_query or not matched_text:
            return normalized_query

        pattern = (
            rf"(?<!\w)"
            f"{re.escape(matched_text)}"
            rf"(?!\w)"
        )

        return re.sub(
            pattern,
            " ",
            normalized_query,
            count=1,
        )

    # ==========================================================
    # PHASE 5C — ANCHOR INTENT
    # ==========================================================

    def _has_explicit_anchor_intent(
        self,
        query: str,
    ) -> bool:

        normalized_query = (
            self._normalize_text(query)
        )

        anchor_intent_patterns = [
            r"\bmirip dengan\b",
            r"\bmirip seperti\b",
            r"\banime yang mirip\b",
            r"\brekomendasikan anime yang mirip\b",
            r"\brekomendasikan anime mirip\b",
            r"\bsaya suka\b",
            r"\baku suka\b",
            r"\bsuka anime\b",
            r"\bsimilar to\b",
            r"\bsimilar anime to\b",
        ]

        return any(
            re.search(
                pattern,
                normalized_query,
            )
            for pattern in anchor_intent_patterns
        )

    # ==========================================================
    # PHASE 5C — ANCHOR DETECTION
    # ==========================================================

    def _find_anchor(
        self,
        query: str,
    ) -> Optional[Dict[str, Any]]:

        normalized_query = self._normalize_text(
            query
        )

        if not normalized_query:
            return None

        if not self._has_explicit_anchor_intent(
            normalized_query
        ):
            return None

        matches = []

        for anchor in self.title_anchors:

            candidate = anchor[
                "normalized_title"
            ]

            display_title = anchor[
                "title"
            ]

            mal_id = anchor[
                "mal_id"
            ]

            if not candidate:
                continue

            # --------------------------------------------------
            # FULL TITLE
            # --------------------------------------------------

            full_pattern = (
                rf"(?<!\w)"
                f"{re.escape(candidate)}"
                rf"(?!\w)"
            )

            if re.search(
                full_pattern,
                normalized_query,
            ):

                matches.append(
                    {
                        "mal_id": mal_id,
                        "title": display_title,
                        "match_type": "full",
                        "matched_text": candidate,
                        "normalized_title": candidate,
                    }
                )

                continue

            # --------------------------------------------------
            # ROOT TITLE
            # --------------------------------------------------

            root = self._title_root(
                candidate
            )

            if not root or len(root) < 3:
                continue

            root_pattern = (
                rf"(?<!\w)"
                f"{re.escape(root)}"
                rf"(?!\w)"
            )

            if re.search(
                root_pattern,
                normalized_query,
            ):

                matches.append(
                    {
                        "mal_id": mal_id,
                        "title": display_title,
                        "match_type": "root",
                        "matched_text": root,
                        "normalized_title": candidate,
                        "root": root,
                    }
                )

        if not matches:
            return None

        def match_priority(item):

            type_priority = {
                "full": 2,
                "root": 1,
            }

            title = item[
                "normalized_title"
            ]

            root = item.get(
                "root",
                self._title_root(title),
            )

            suffix_length = max(
                0,
                len(title) - len(root),
            )

            token_count = len(
                title.split()
            )

            return (
                type_priority.get(
                    item["match_type"],
                    0,
                ),
                -suffix_length,
                -token_count,
                len(
                    item["matched_text"]
                ),
            )

        matches.sort(
            key=match_priority,
            reverse=True,
        )

        best = matches[0]

        print(
            f"[ANCHOR MATCH] "
            f"'{best['matched_text']}' "
            f"-> {best['title']} "
            f"(mal_id={best['mal_id']}, "
            f"type={best['match_type']})"
        )

        return {
            "mal_id": best["mal_id"],
            "title": best["title"],
        }

    # Compatibility aliases
    def _find_anchor_mal_id(
        self,
        query: str,
    ) -> Optional[int]:

        anchor = self._find_anchor(
            query
        )

        if anchor is None:
            return None

        return anchor["mal_id"]

    def _extract_title_root(
        self,
        title: Any,
    ) -> str:

        return self._title_root(title)

    # ==========================================================
    # PHASE 5C — FRANCHISE
    # ==========================================================

    @staticmethod
    def _title_root(
        title: Any,
    ) -> str:

        if title is None:
            return ""

        value = str(
            title
        ).strip().lower()

        if not value:
            return ""

        value = re.sub(
            r"\([^)]*\)",
            "",
            value,
        )

        value = re.split(
            r"\s*:\s*",
            value,
            maxsplit=1,
        )[0]

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        return value

    def _is_same_franchise(
        self,
        anchor_mal_id: Any,
        candidate: Dict[str, Any],
    ) -> bool:

        candidate_metadata = candidate.get(
            "metadata",
            {},
        )

        candidate_mal_id = candidate.get(
            "mal_id"
        )

        if candidate_mal_id == anchor_mal_id:
            return True

        anchor_document = (
            self.document_lookup.get(
                anchor_mal_id
            )
        )

        if anchor_document is None:
            return False

        anchor_metadata = (
            anchor_document.get(
                "metadata",
                {},
            )
        )

        anchor_titles = []

        for field in (
            "title",
            "title_english",
        ):

            value = anchor_metadata.get(
                field
            )

            if value:

                normalized = (
                    self._normalize_text(
                        str(value)
                    )
                )

                if normalized:
                    anchor_titles.append(
                        normalized
                    )

        candidate_titles = []

        for field in (
            "title",
            "title_english",
        ):

            value = candidate_metadata.get(
                field
            )

            if value:

                normalized = (
                    self._normalize_text(
                        str(value)
                    )
                )

                if normalized:
                    candidate_titles.append(
                        normalized
                    )

        if not anchor_titles or not candidate_titles:
            return False

        anchor_roots = {
            self._title_root(title)
            for title in anchor_titles
            if self._title_root(title)
        }

        candidate_roots = {
            self._title_root(title)
            for title in candidate_titles
            if self._title_root(title)
        }

        if anchor_roots & candidate_roots:
            return True

        # Phrase matching
        for anchor_root in anchor_roots:

            if len(anchor_root) < 4:
                continue

            pattern = (
                rf"(?<!\w)"
                f"{re.escape(anchor_root)}"
                rf"(?!\w)"
            )

            for candidate_title in candidate_titles:

                if re.search(
                    pattern,
                    candidate_title,
                ):
                    return True

        return False

    def _apply_anchor_exclusion(
        self,
        candidates: List[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if anchor is None:
            return candidates

        anchor_mal_id = anchor[
            "mal_id"
        ]

        filtered_candidates = []
        excluded_count = 0

        for candidate in candidates:

            if self._is_same_franchise(
                anchor_mal_id,
                candidate,
            ):

                excluded_count += 1

            else:

                filtered_candidates.append(
                    candidate
                )

        print(
            f"[INFO] Anchor franchise exclusion: "
            f"{excluded_count} candidates removed"
        )

        return filtered_candidates

    # ==========================================================
    # PHASE 5C — ANCHOR RETRIEVAL
    # ==========================================================

    def _retrieve_anchor_candidates(
        self,
        anchor: Dict[str, Any],
        candidate_k: int,
    ) -> List[Dict[str, Any]]:

        if not isinstance(
            anchor,
            dict,
        ):
            raise TypeError(
                "anchor harus berupa dictionary."
            )

        if not isinstance(
            candidate_k,
            int,
        ):
            raise TypeError(
                "candidate_k harus berupa integer."
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k harus lebih besar dari 0."
            )

        candidate_k = min(
            candidate_k,
            self.index.ntotal,
        )

        anchor_mal_id = anchor.get(
            "mal_id"
        )

        if anchor_mal_id is None:
            raise KeyError(
                "Anchor tidak memiliki mal_id."
            )

        anchor_document = (
            self.document_lookup.get(
                anchor_mal_id
            )
        )

        if anchor_document is None:
            raise KeyError(
                "Dokumen anchor tidak ditemukan: "
                f"mal_id={anchor_mal_id}"
            )

        print(
            f"[ANCHOR RETRIEVAL] "
            f"Menggunakan document anchor: "
            f"{anchor_document['metadata'].get('title', anchor['mal_id'])} "
            f"(mal_id={anchor_mal_id})"
        )

        anchor_embedding = (
            self._encode_anchor_document(
                anchor
            )
        )

        return self._search_faiss(
            anchor_embedding,
            candidate_k,
        )

    # ==========================================================
    # PHASE 5C — RERANKING
    # ==========================================================

    @staticmethod
    def _normalize_mal_score(
        value: Any,
    ) -> float:

        if value is None:
            return 0.0

        try:
            score = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        score = max(
            0.0,
            min(score, 10.0),
        )

        return score / 10.0

    def _rerank(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not candidates:
            return []

        similarities = np.array(
            [
                candidate["similarity"]
                for candidate in candidates
            ],
            dtype=np.float32,
        )

        min_similarity = float(
            similarities.min()
        )

        max_similarity = float(
            similarities.max()
        )

        similarity_range = (
            max_similarity
            - min_similarity
        )

        for candidate in candidates:

            semantic_score = float(
                candidate["similarity"]
            )

            if similarity_range > 1e-12:

                normalized_similarity = (
                    semantic_score
                    - min_similarity
                ) / similarity_range

            else:

                normalized_similarity = 1.0

            normalized_mal_score = (
                self._normalize_mal_score(
                    candidate["metadata"].get(
                        "score"
                    )
                )
            )

            final_score = (
                self.semantic_weight
                * normalized_similarity
                +
                self.score_weight
                * normalized_mal_score
            )

            candidate[
                "normalized_similarity"
            ] = float(
                normalized_similarity
            )

            candidate[
                "normalized_mal_score"
            ] = float(
                normalized_mal_score
            )

            candidate[
                "rerank_score"
            ] = float(
                final_score
            )

        candidates.sort(
            key=lambda item: (
                item["rerank_score"],
                item["similarity"],
            ),
            reverse=True,
        )

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):

            candidate[
                "final_rank"
            ] = rank

        return candidates

    # ==========================================================
    # PHASE 5C — COMPLETE RETRIEVAL
    # ==========================================================

    def retrieve_filtered(
        self,
        query: str,
        top_k: int | None = None,
    ) -> Dict[str, Any]:

        if not isinstance(query, str):
            raise TypeError("query harus berupa string.")

        query = query.strip()
        if not query:
            raise ValueError("query tidak boleh kosong.")

        if top_k is None:
            top_k = self.retrieval_top_k

        if not isinstance(top_k, int):
            raise TypeError("top_k harus berupa integer.")

        if top_k <= 0:
            raise ValueError("top_k harus lebih besar dari 0.")

        # ------------------------------------------------------
        # 1. Intent / entity resolution
        # ------------------------------------------------------
        # Similarity anchor tetap terpisah dari factual entity resolution.
        anchor = self._find_anchor(query)

        factual_entity = None
        if anchor is None:
            factual_entity = self._resolve_factual_entity(query)

        # ------------------------------------------------------
        # 2. Attribute filters
        # ------------------------------------------------------
        # Jika title sudah ter-resolve secara exact, token di dalam title
        # tidak boleh dianggap sebagai attribute filter.
        filter_query = query
        if factual_entity is not None:
            filter_query = self._remove_matched_title_span(
                query,
                factual_entity,
            )

        filters = self._detect_attribute_filters(filter_query)

        # ------------------------------------------------------
        # 3. Factual exact-title retrieval
        # ------------------------------------------------------
        if factual_entity is not None:
            mal_id = factual_entity["mal_id"]
            document = self.document_lookup.get(mal_id)

            if document is not None:
                factual_candidate = {
                    "rank": 1,
                    "mal_id": mal_id,
                    "similarity": 1.0,
                    "text": document["text"],
                    "metadata": document["metadata"],
                }

                before_filtering = 1
                filtered = self._apply_attribute_filters(
                    [factual_candidate],
                    filters,
                )
                after_filtering = len(filtered)
                after_anchor_exclusion = after_filtering

                final_results = []
                if filtered:
                    result = filtered[0]
                    result["final_rank"] = 1
                    result["exact_match"] = True
                    result["retrieval_score"] = 1.0
                    final_results = [result]

                has_hard_filters = self._has_hard_filters(filters)

                print()
                print("[FACTUAL EXACT TITLE] Retrieval")
                print(f"[QUERY] {query}")
                print(f"[ENTITY] {factual_entity}")
                print(f"[FILTER QUERY] {filter_query}")
                print(f"[FILTERS] {filters}")
                print(f"[INFO] After attribute filter = {after_filtering}")
                print(f"[INFO] Final results = {len(final_results)}")

                return {
                    "query": query,
                    "filters": filters,
                    "anchor": None,
                    "factual_entity": factual_entity,
                    "retrieval_mode": "factual_exact_title",
                    "candidate_k": 1,
                    "hard_filters_active": has_hard_filters,
                    "candidate_count": before_filtering,
                    "after_attribute_filter": after_filtering,
                    "after_anchor_exclusion": after_anchor_exclusion,
                    "results": final_results,
                }

        # ------------------------------------------------------
        # 4. Standard candidate pool
        # ------------------------------------------------------
        candidate_k = self._determine_candidate_k(
            filters,
            anchor=anchor,
        )
        has_hard_filters = self._has_hard_filters(filters)

        # ------------------------------------------------------
        # 5. Semantic retrieval
        # ------------------------------------------------------
        if anchor is not None:
            candidates = self._retrieve_anchor_candidates(
                anchor,
                candidate_k,
            )
            retrieval_mode = "anchor_document"
        else:
            candidates = self._retrieve_candidates(
                query,
                candidate_k,
            )
            retrieval_mode = "user_query"

        before_filtering = len(candidates)

        # ------------------------------------------------------
        # 6. Hard filtering
        # ------------------------------------------------------
        filtered = self._apply_attribute_filters(
            candidates,
            filters,
        )
        after_filtering = len(filtered)

        # ------------------------------------------------------
        # 7. Anchor/franchise exclusion
        # ------------------------------------------------------
        filtered = self._apply_anchor_exclusion(
            filtered,
            anchor,
        )
        after_anchor_exclusion = len(filtered)

        # ------------------------------------------------------
        # 8. Reranking
        # ------------------------------------------------------
        reranked = self._rerank(filtered)

        # ------------------------------------------------------
        # 9. Final top-k
        # ------------------------------------------------------
        final_results = reranked[:top_k]

        # ------------------------------------------------------
        # 10. Diagnostic
        # ------------------------------------------------------
        print()
        print("[PHASE 5C] Retrieval pipeline")
        print(f"[QUERY] {query}")
        print(f"[RETRIEVAL MODE] {retrieval_mode}")
        print("[FILTERS]")
        print(f"  genres      = {filters['genres']}")
        print(f"  themes      = {filters['themes']}")
        print(f"  min_score   = {filters['min_score']}")
        print(f"  max_score   = {filters['max_score']}")
        print(f"  min_year    = {filters['min_year']}")
        print(f"  max_year    = {filters['max_year']}")
        print(f"  type        = {filters['type']}")
        print(f"  studios     = {filters['studios']}")
        print(f"[INFO] Hard filters active = {has_hard_filters}")
        print(f"[INFO] Candidate K = {candidate_k}")
        print(f"[INFO] FAISS candidates = {before_filtering}")
        print(f"[INFO] After attribute filter = {after_filtering}")
        print(f"[INFO] Anchor = {anchor}")
        print(f"[INFO] After anchor exclusion = {after_anchor_exclusion}")
        print(f"[INFO] Final results = {len(final_results)}")

        return {
            "query": query,
            "filters": filters,
            "anchor": anchor,
            "factual_entity": factual_entity,
            "retrieval_mode": retrieval_mode,
            "candidate_k": candidate_k,
            "hard_filters_active": has_hard_filters,
            "candidate_count": before_filtering,
            "after_attribute_filter": after_filtering,
            "after_anchor_exclusion": after_anchor_exclusion,
            "results": final_results,
        }

    # ==========================================================
    # PHASE 7 — MULTI-TURN
    # ==========================================================

    def is_followup(
        self,
        query: str,
    ) -> bool:

        if not isinstance(
            query,
            str,
        ):
            return False

        normalized = self._normalize_text(
            query
        ).strip()

        if not normalized:
            return False

        words = normalized.split()

        # Scope penelitian:
        # follow-up hanya 2 turn.
        if len(words) > 15:
            return False

        # ----------------------------------------------------------
        # Phrase-aware matching
        # ----------------------------------------------------------
        #
        # Tambahkan boundary agar hint tidak terdeteksi sebagai
        # substring dari kata lain.
        #
        # Contoh:
        #   "ova" tidak digunakan lagi sebagai hint tunggal.
        #   Dengan demikian tidak ada false-positive seperti
        #   substring "ova" pada kata lain.
        #
        normalized_padded = f" {normalized} "

        for hint in FOLLOWUP_HINTS:
            hint_normalized = self._normalize_text(
                hint
            ).strip()

            if not hint_normalized:
                continue

            # Phrase boundary:
            # memastikan hint merupakan token/frasa utuh.
            if (
                f" {hint_normalized} "
                in normalized_padded
            ):
                return True

        return False

    def build_effective_query(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build effective query for 2-turn refinement.

        Only the immediately preceding user turn is used.
        This intentionally does NOT implement cumulative
        3+ turn memory.
        """

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query harus berupa string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query tidak boleh kosong."
            )

        if not history:
            return query

        if not self.is_followup(query):
            return query

        previous_user_query = None

        for message in reversed(history):

            if not isinstance(
                message,
                dict,
            ):
                continue

            role = message.get(
                "role"
            )

            content = message.get(
                "content"
            )

            if (
                role == "user"
                and isinstance(content, str)
                and content.strip()
            ):

                previous_user_query = (
                    content.strip()
                )

                break

        if not previous_user_query:
            return query

        effective_query = (
            f"{previous_user_query}. "
            f"{query}"
        )

        print(
            "[MULTI-TURN] Effective query:"
        )

        print(
            f"  previous = {previous_user_query}"
        )

        print(
            f"  current  = {query}"
        )

        print(
            f"  effective = {effective_query}"
        )

        return effective_query

    # ==========================================================
    # SLM — LOAD
    # ==========================================================

    def load_llm(
        self,
        model_name: Optional[str] = None,
        quantize: bool = True,
    ):
        """
        Load Llama-3.2-3B-Instruct.

        Intended for GPU environments such as Kaggle T4.
        """

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "load_llm() membutuhkan CUDA "
                "untuk konfigurasi SLM ini."
            )

        if model_name is None:

            model_name = (
                self.config["llm"].get(
                    "model_name",
                    "meta-llama/Llama-3.2-3B-Instruct",
                )
            )

        print(
            f"[INFO] Memuat SLM: {model_name}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto",
        }

        if quantize:

            compute_dtype = (
                torch.float16
            )

            quantization_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            )

            model_kwargs[
                "quantization_config"
            ] = quantization_config

        else:

            model_kwargs[
                "torch_dtype"
            ] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        self.llm = model
        self.tokenizer = tokenizer
        self.text_generation_pipeline = (
            text_pipeline
        )

        print(
            "[OK] SLM berhasil dimuat"
        )

        return self.llm

    # ==========================================================
    # OPTIONAL GGUF COMPATIBILITY
    # ==========================================================

    def load_llm_gguf(
        self,
        *args,
        **kwargs,
    ):
        """
        Optional legacy compatibility method.

        GGUF is not required for the target ZeroGPU
        deployment architecture.
        """

        raise NotImplementedError(
            "GGUF/llama.cpp bukan backend deployment "
            "target AniRAG-v2. Gunakan load_llm()."
        )

    # ==========================================================
    # GENERATION
    # ==========================================================

    def generate(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        use_retrieval: bool = True,
        use_enrichment: bool = False,
        pre_retrieved: Optional[
            Dict[str, Any]
        ] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Unified generation interface.

        Conditions:

            A:
                SLM only

            B:
                SLM + RAG

            C:
                SLM + RAG + enrichment

        Guardrail is executed before retrieval.

        Multi-turn effective query is constructed
        using only the immediately previous user turn.
        """

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query harus berupa string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query tidak boleh kosong."
            )

        if top_k is None:
            top_k = self.retrieval_top_k

        if not isinstance(
            top_k,
            int,
        ):
            raise TypeError(
                "top_k harus berupa integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k harus lebih besar dari 0."
            )

        # ------------------------------------------------------
        # 1. GUARDRAIL FIRST
        # ------------------------------------------------------

        refusal = guard_query(
            query
        )

        if refusal is not None:

            print(
                "[GUARDRAIL] Query ditolak."
            )

            return {
                "query": query,
                "effective_query": query,
                "condition": (
                    "blocked"
                ),
                "blocked": True,
                "refusal": refusal,
                "response": refusal,
                "results": [],
                "context": "",
            }

        # ------------------------------------------------------
        # 2. MULTI-TURN QUERY
        # ------------------------------------------------------

        effective_query = (
            self.build_effective_query(
                query,
                history=history,
            )
        )

        # ------------------------------------------------------
        # 3. RETRIEVAL
        # ------------------------------------------------------

        retrieval_output = None

        if use_retrieval:

            if pre_retrieved is not None:

                retrieval_output = (
                    pre_retrieved
                )

            else:

                retrieval_output = (
                    self.retrieve_filtered(
                        effective_query,
                        top_k=top_k,
                    )
                )

        # ------------------------------------------------------
        # 4. BUILD CONTEXT
        # ------------------------------------------------------

        context = ""

        if retrieval_output is not None:

            context = self.build_context(
                retrieval_output.get(
                    "results",
                    [],
                )
            )

        # ------------------------------------------------------
        # 5. CONDITION
        # ------------------------------------------------------

        if not use_retrieval:

            condition = "A"

        elif use_enrichment:

            condition = "C"

        else:

            condition = "B"

        # ------------------------------------------------------
        # 6. ENRICHMENT
        # ------------------------------------------------------

        enrichment_data = None

        if (
            use_enrichment
            and retrieval_output is not None
        ):

            enrichment_data = (
                self.enrich(
                    retrieval_output.get(
                        "results",
                        [],
                    )
                )
            )

        # ------------------------------------------------------
        # 7. ENSURE LLM
        # ------------------------------------------------------

        if self.text_generation_pipeline is None:

            self.load_llm()

        # ------------------------------------------------------
        # 8. SYSTEM PROMPT
        # ------------------------------------------------------

        if use_retrieval:

            system_prompt = (
                SYSTEM_PROMPT_RAG.format(
                    context=context
                )
            )

            if enrichment_data:

                system_prompt += (
                    "\n\nDATA ENRICHMENT:\n"
                    + str(enrichment_data)
                )

        else:

            system_prompt = (
                SYSTEM_PROMPT_BASELINE
            )

        user_prompt = (
            PROMPT_TEMPLATE.format(
                query=effective_query
            )
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        # ------------------------------------------------------
        # 9. CHAT TEMPLATE
        # ------------------------------------------------------

        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):

            prompt_text = (
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        else:

            prompt_text = (
                f"{system_prompt}\n\n"
                f"{user_prompt}"
            )

        # ------------------------------------------------------
        # 10. GENERATION
        # ------------------------------------------------------

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "return_full_text": False,
        }

        terminators = []

        if self.tokenizer.eos_token_id is not None:
            terminators.append(
                self.tokenizer.eos_token_id
            )

        eot_id = (
            self.tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"
            )
        )

        if (
            eot_id is not None
            and eot_id >= 0
            and eot_id not in terminators
        ):
            terminators.append(
                eot_id
            )

        if terminators:

            generation_kwargs[
                "eos_token_id"
            ] = terminators

        outputs = (
            self.text_generation_pipeline(
                prompt_text,
                **generation_kwargs,
            )
        )

        generated_text = (
            outputs[0]["generated_text"]
            if outputs
            else ""
        )

        generated_text = (
            generated_text.strip()
        )

        # ------------------------------------------------------
        # 11. RESULT
        # ------------------------------------------------------

        return {
            "query": query,
            "effective_query": effective_query,
            "condition": condition,
            "blocked": False,
            "refusal": None,
            "response": generated_text,
            "results": (
                retrieval_output.get(
                    "results",
                    [],
                )
                if retrieval_output
                else []
            ),
            "retrieval": retrieval_output,
            "context": context,
            "enrichment": enrichment_data,
        }

    # ==========================================================
    # CONTEXT
    # ==========================================================

    def build_context(
        self,
        results: List[Dict[str, Any]],
    ) -> str:

        if not results:
            return (
                "Tidak ada hasil retrieval."
            )

        context_parts = []

        for rank, result in enumerate(
            results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            title = metadata.get(
                "title",
                "Unknown",
            )

            title_english = metadata.get(
                "title_english"
            )

            anime_type = metadata.get(
                "type"
            )

            episodes = metadata.get(
                "episodes"
            )

            score = metadata.get(
                "score"
            )

            genres = metadata.get(
                "genres"
            )

            themes = metadata.get(
                "themes"
            )

            year = metadata.get(
                "year"
            )

            studios = metadata.get(
                "studios"
            )

            synopsis = metadata.get(
                "synopsis"
            )

            block = (
                f"[Anime {rank}]\n"
                f"Title: {title}\n"
                f"English Title: {title_english}\n"
                f"Type: {anime_type}\n"
                f"Episodes: {episodes}\n"
                f"Score: {score}\n"
                f"Genres: {genres}\n"
                f"Themes: {themes}\n"
                f"Year: {year}\n"
                f"Studios: {studios}\n"
                f"Synopsis: {synopsis}\n"
            )

            context_parts.append(
                block
            )

        return "\n".join(
            context_parts
        )

    # ==========================================================
    # ENRICHMENT
    # ==========================================================

    def enrich(
        self,
        results: List[Dict[str, Any]],
    ):
        """
        Optional enrichment hook.

        The base Phase 5C / RAG path does not depend
        on enrichment.

        If a Jikan client is attached by the project,
        it may be used by an external implementation.
        """

        if not results:
            return None

        # Intentionally conservative.
        #
        # Phase 5C and primary RAG evaluation must remain
        # independent from external enrichment.
        return None


# ==============================================================
# SMOKE TEST
# ==============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "AniRAG-v2 — UNIFIED RAG PIPELINE"
    )
    print(
        "Phase 5A + 5B + 5C + SLM + Guardrail + Multi-turn"
    )
    print("=" * 70)

    pipeline = RagPipeline()

    # ----------------------------------------------------------
    # FACTUAL / ATTRIBUTE INTENT REGRESSION TEST
    # ----------------------------------------------------------

    print("=" * 70)
    print("[FACTUAL / ATTRIBUTE INTENT REGRESSION TEST]")
    print("=" * 70)

    attribute_query = (
        "Rekomendasikan anime dengan tema Historical."
    )

    factual_intent = pipeline._has_factual_intent(
        attribute_query
    )

    filters = pipeline._detect_attribute_filters(
        attribute_query
    )

    print(
        f"[QUERY] {attribute_query}"
    )

    print(
        f"[FACTUAL INTENT] {factual_intent}"
    )

    print(
        f"[FILTERS] {filters}"
    )

    if factual_intent is False:
        print("[PASS] Query tidak dianggap sebagai factual entity query.")
    else:
        print("[FAIL] Query salah dianggap sebagai factual entity query.")

    if "historical" in filters.get("themes", []):
        print("[PASS] Theme Historical berhasil terdeteksi.")
    else:
        print("[FAIL] Theme Historical tidak terdeteksi.")

    attribute_result = pipeline.retrieve_filtered(
        attribute_query,
        top_k=5,
    )

    print(
        f"[RETRIEVAL MODE] "
        f"{attribute_result['retrieval_mode']}"
    )

    print(
        f"[RETRIEVAL FILTERS] "
        f"{attribute_result['filters']}"
    )

    print("[RESULT] Top-5:")

    for item in attribute_result["results"]:
        metadata = item["metadata"]

        print(
            f"- {item['mal_id']} | "
            f"{metadata.get('title')} | "
            f"themes={metadata.get('themes')}"
        )

    # ----------------------------------------------------------
    # Numeric / Temporal Attribute Filter Test
    # ----------------------------------------------------------

    print("=" * 70)
    print("[NUMERIC / TEMPORAL ATTRIBUTE FILTER TEST]")
    print("=" * 70)

    numeric_temporal_tests = [
        "anime dengan score minimal 8",
        "anime dengan score di atas 8",
        "anime dengan score maksimal 8",
        "anime dengan score di bawah 8",
        "anime tahun 2020",
        "anime tahun 2020 ke atas",
        "anime tahun 2020 ke bawah",
        "anime setelah 2018",
        "anime sebelum 2010",
    ]

    for query in numeric_temporal_tests:

        filters = pipeline._detect_attribute_filters(
            query
        )

        print(
            f"[QUERY] {query}"
        )

        print(
            f"[FILTERS] {filters}"
        )

        print()

    # ----------------------------------------------------------
    # Numeric / Temporal HARD FILTER VALIDATION
    # ----------------------------------------------------------

    print("=" * 70)
    print("[NUMERIC / TEMPORAL HARD FILTER VALIDATION]")
    print("=" * 70)

    hard_filter_tests = [
        (
            "Rekomendasikan anime dengan score minimal 8",
            lambda m: (
                m.get("score") is not None
                and float(m["score"]) >= 8
            ),
        ),
        (
            "Rekomendasikan anime dengan score maksimal 8",
            lambda m: (
                m.get("score") is not None
                and float(m["score"]) <= 8
            ),
        ),
        (
            "Rekomendasikan anime tahun 2020 ke atas",
            lambda m: (
                m.get("year") is not None
                and int(float(m["year"])) >= 2020
            ),
        ),
        (
            "Rekomendasikan anime tahun 2020 ke bawah",
            lambda m: (
                m.get("year") is not None
                and int(float(m["year"])) <= 2020
            ),
        ),
        (
            "Rekomendasikan anime setelah 2018",
            lambda m: (
                m.get("year") is not None
                and int(float(m["year"])) > 2018
            ),
        ),
        (
            "Rekomendasikan anime sebelum 2010",
            lambda m: (
                m.get("year") is not None
                and int(float(m["year"])) < 2010
            ),
        ),
    ]

    for query, validator in hard_filter_tests:

        output = pipeline.retrieve_filtered(
            query,
            top_k=5,
        )

        results = output["results"]

        violations = []

        for item in results:

            metadata = item["metadata"]

            try:
                valid = validator(metadata)
            except (
                TypeError,
                ValueError,
            ):
                valid = False

            if not valid:
                violations.append(
                    (
                        item["mal_id"],
                        metadata.get("title"),
                        metadata.get("score"),
                        metadata.get("year"),
                    )
                )

        if violations:
            print(f"[FAIL] {query}")

            for violation in violations:
                print(
                    f"  - mal_id={violation[0]} | "
                    f"title={violation[1]} | "
                    f"score={violation[2]} | "
                    f"year={violation[3]}"
                )

        else:
            print(
                f"[PASS] {query} "
                f"({len(results)} results)"
            )

    # ----------------------------------------------------------
    # Guardrail system test melalui pipeline.generate()
    # ----------------------------------------------------------

    print("=" * 70)
    print("[GUARDRAIL SYSTEM TEST — generate()]")
    print("=" * 70)

    blocked_query = "Rekomendasikan anime hentai"

    result = pipeline.generate(blocked_query)

    print("[QUERY]", blocked_query)
    print("[RESULT]", result)

    # ----------------------------------------------------------
    # Phase 5C retrieval smoke test
    # ----------------------------------------------------------

    test_query = (
        "Saya suka Naruto, "
        "rekomendasikan anime yang mirip"
    )

    test_query = (
        "Saya suka Naruto, "
        "rekomendasikan anime yang mirip"
    )

    output = pipeline.retrieve_filtered(
        test_query,
        top_k=5,
    )

    print()
    print("[RESULT] Final Top-K:")
    print("-" * 70)

    for result in output["results"]:

        metadata = result["metadata"]

        title = metadata.get(
            "title",
            "Unknown",
        )

        print(
            f"{result['final_rank']}. "
            f"{title}"
        )

        print(
            f"   mal_id="
            f"{result['mal_id']}"
        )

        print(
            f"   similarity="
            f"{result['similarity']:.4f}"
        )

        print(
            f"   MAL score="
            f"{metadata.get('score')}"
        )

        print(
            f"   rerank_score="
            f"{result['rerank_score']:.4f}"
        )

        print()

    # ----------------------------------------------------------
    # Multi-turn smoke test
    # ----------------------------------------------------------

    print("=" * 70)
    print("[MULTI-TURN SMOKE TEST]")
    print("=" * 70)

    history = [
        {
            "role": "user",
            "content": (
                "Rekomendasikan anime action"
            ),
        }
    ]

    followup = (
        "yang skornya di atas 8"
    )

    effective_query = (
        pipeline.build_effective_query(
            followup,
            history,
        )
    )

    print(
        f"[FOLLOW-UP] {followup}"
    )

    print(
        f"[EFFECTIVE] {effective_query}"
    )

    # ----------------------------------------------------------
    # Follow-up detection test
    # ----------------------------------------------------------

    print("\n[MULTI-TURN FOLLOW-UP TEST]")

    followup_tests = [
        ("yang skornya di atas 8", True),
        ("yang ratingnya di bawah 7", True),
        ("yang lain", True),
        ("tadi yang mana?", True),
        ("anime tahun 2023", False),
        ("rekomendasikan anime genre action", False),
        ("anime dengan tema isekai", False),
        ("anime dari studio MAPPA", False),
        ("jenis anime apa?", False),
        ("anime TV terbaik", False),
        ("anime movie terbaik", False),
        ("anime OVA terbaik", False),
        ("anime ONA terbaik", False),
        (
            "tolong rekomendasikan anime action dengan cerita "
            "yang sangat menarik dan karakter utama yang kuat "
            "serta memiliki perkembangan karakter yang bagus",
            False
        ),
    ]

    for query, expected in followup_tests:
        actual = pipeline.is_followup(query)

        status = (
            "PASS"
            if actual == expected
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"expected={expected} "
            f"actual={actual} "
            f"query={query}"
        )


    # ----------------------------------------------------------
    # Guardrail smoke test
    # ----------------------------------------------------------

    print("=" * 70)
    print("[GUARDRAIL SMOKE TEST]")
    print("=" * 70)

    guardrail_tests = [
        "Rekomendasikan anime action",
        "Rekomendasikan anime hentai",
        "Bagaimana cara membuat program Python?",
        "Berikan link streaming gratis Naruto",
    ]

    for test in guardrail_tests:

        refusal = guard_query(test)

        status = (
            "BLOCKED"
            if refusal
            else "ALLOWED"
        )

        print(
            f"[{status}] {test}"
        )

    print("=" * 70)
    print(
        "UNIFIED RAG PIPELINE SMOKE TEST SELESAI"
    )
    print("=" * 70)
