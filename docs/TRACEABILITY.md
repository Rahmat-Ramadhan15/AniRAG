# THEORY → CODE → EXPERIMENT → UI TRACEABILITY

> **Status:** DRAFT — Phase 15
> **Versi:** Sementara, sebelum audit `app.py` pada Phase 16
> **Tujuan:** Mendokumentasikan keterlacakan antara konsep teori, implementasi kode, fungsi, evidence eksperimen, dan perilaku aplikasi.

---

## 1. Tujuan Traceability

Dokumen ini digunakan untuk menunjukkan hubungan antara konsep teoritis yang digunakan dalam sistem AniRAG-v2 dengan implementasi aktual pada source code dan evidence eksperimen.

Setiap konsep harus dapat ditelusuri melalui rantai:

**Konsep Teori → Implementasi → File → Function → Evidence → UI**

Pada Phase 15, source code yang telah diaudit meliputi:

- `src/preprocess.py`
- `src/build_index.py`
- `src/rag_pipeline.py`
- `src/guardrails.py`

Bagian UI yang membutuhkan `app.py` belum difinalkan dan akan dilengkapi pada **Phase 16**.

---

# 2. Traceability Matrix

| Konsep Teori              | Implementasi                          | File                  | Function                                                    | Evidence                                                                                     | UI                       |
| ------------------------- | ------------------------------------- | --------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------ |
| Document Representation   | Structured anime document             | `src/preprocess.py`   | `build_document()`                                          | Setiap anime direpresentasikan sebagai dokumen teks terstruktur berisi metadata dan synopsis | —                        |
| Corpus Construction       | JSONL document corpus                 | `src/preprocess.py`   | `build_documents()`                                         | 15.966 dokumen dibangun dari dataset final                                                   | —                        |
| Dataset Validation        | Validasi schema dan data              | `src/preprocess.py`   | `validate_dataset()`                                        | 15.966 rows, 12 kolom, `mal_id` unik, required fields tervalidasi                            | —                        |
| Content Safety Validation | Validasi blocked tags                 | `src/preprocess.py`   | `validate_dataset()`                                        | `genres/themes` diperiksa terhadap `hentai` dan `erotica`                                    | —                        |
| Embedding                 | Sentence Transformer                  | `src/build_index.py`  | `build_index()`                                             | `sentence-transformers/all-MiniLM-L6-v2`, dimensi 384                                        | —                        |
| Embedding Normalization   | L2-normalized embeddings              | `src/build_index.py`  | `build_index()`                                             | Norm embedding divalidasi mendekati 1.0                                                      | —                        |
| Vector Index              | FAISS `IndexFlatIP`                   | `src/build_index.py`  | `build_index()`                                             | FAISS index dibangun dengan dimensi embedding 384                                            | —                        |
| ID Mapping                | Vector index ↔ `mal_id` mapping       | `src/build_index.py`  | `build_index()`                                             | Jumlah mapping divalidasi terhadap jumlah vector                                             | —                        |
| Retrieval                 | Semantic similarity search            | `src/rag_pipeline.py` | `_search_faiss()`, `retrieve_filtered()`                    | Retrieval experiment Phase 9; Recall/MRR/P@K                                                 | Recommendations          |
| Attribute Filtering       | Metadata hard filtering               | `src/rag_pipeline.py` | `_detect_attribute_filters()`, `_apply_attribute_filters()` | Phase 9 attribute evaluation                                                                 | Filtered recommendations |
| Anchor Retrieval          | Retrieval berdasarkan anime referensi | `src/rag_pipeline.py` | `_find_anchor()`, `_retrieve_anchor_candidates()`           | Phase 9 similarity evaluation                                                                | Similar recommendations  |
| Franchise Exclusion       | Same-franchise exclusion              | `src/rag_pipeline.py` | `_apply_anchor_exclusion()`                                 | Candidate filtering pada similarity retrieval                                                | —                        |
| Reranking                 | Semantic similarity + MAL score       | `src/rag_pipeline.py` | `_rerank()`                                                 | Ranking menggunakan bobot 0.6 semantic + 0.4 MAL score                                       | Ranked recommendations   |
| RAG                       | Context augmentation                  | `src/rag_pipeline.py` | `build_context()`, `generate()`                             | Phase 10 A/B experiment; B menggunakan retrieval + context                                   | Grounded response        |
| SLM                       | Llama 3.2 3B Instruct                 | `src/rag_pipeline.py` | `load_llm()`, `generate()`                                  | Phase 10 generation experiment                                                               | Generated response       |
| Quantized SLM             | 4-bit BitsAndBytes                    | `src/rag_pipeline.py` | `load_llm()`                                                | 4-bit NF4 configuration dengan `device_map="auto"`                                           | —                        |
| Guardrail                 | Explicit content hard-block           | `src/guardrails.py`   | `is_explicit_request()`, `guard_query()`                    | Phase 10 safety evaluation                                                                   | Refusal                  |
| Guardrail                 | Non-anime topic hard-block            | `src/guardrails.py`   | `is_non_anime_topic()`, `guard_query()`                     | Phase 10 safety evaluation                                                                   | Refusal                  |
| Guardrail                 | Streaming/download hard-block         | `src/guardrails.py`   | `is_streaming_request()`, `guard_query()`                   | Phase 10 safety evaluation                                                                   | Refusal                  |
| Guardrail                 | Personal-profile hard-block           | `src/guardrails.py`   | `is_personal_profile_request()`, `guard_query()`            | Implemented; dedicated UAT trigger belum tersedia pada evidence Phase 10                     | Refusal                  |
| Guardrail                 | OOS diagnostic heuristic              | `src/guardrails.py`   | `looks_out_of_scope()`                                      | Digunakan untuk kategori/pelaporan evaluasi; bukan hard-block                                | —                        |
| Multi-turn                | Contextual query construction         | `src/rag_pipeline.py` | `is_followup()`, `build_effective_query()`                  | Phase 9/10 multi-turn test set                                                               | Follow-up response       |
| Multi-turn Scope          | Two-turn context                      | `src/rag_pipeline.py` | `build_effective_query()`                                   | Hanya immediate previous user turn yang digabungkan                                          | Chat                     |
| Poster                    | `image_url` metadata                  | `src/preprocess.py`   | `build_documents()`                                         | `image_url` dipertahankan sebagai metadata, tidak dimasukkan ke embedding text               | **TBD Phase 16**         |
| Poster Rendering          | Application UI                        | `app.py`              | **TBD**                                                     | **Belum diaudit**                                                                            | **TBD Phase 16**         |
| Chat Interface            | Gradio application                    | `app.py`              | **TBD**                                                     | **Belum diaudit**                                                                            | **TBD Phase 16**         |
| Conversation Display      | Chat/history handling                 | `app.py`              | **TBD**                                                     | **Belum diaudit**                                                                            | **TBD Phase 16**         |

---

# 3. Dataset and Document Representation

## 3.1 Final Dataset

Dataset final yang digunakan oleh pipeline yang telah diaudit terdiri dari:

**15.966 rows × 12 columns**

Kolom:

```text
mal_id
title
title_english
type
episodes
score
genres
synopsis
year
themes
studios
image_url
```

Validasi dilakukan pada `src/preprocess.py`.

Function utama:

```text
load_dataset()
validate_dataset()
```

`validate_dataset()` memastikan:

- jumlah baris sesuai baseline;
- schema dan urutan kolom sesuai;
- `mal_id` tidak kosong;
- `mal_id` unik;
- required fields tersedia;
- tidak terdapat tag `hentai` atau `erotica` pada `genres/themes`.

### Catatan penting

`preprocess.py` yang telah diaudit membuktikan **validasi terhadap blocked tags**, tetapi source tersebut tidak membuktikan proses penghapusan/filtering data dilakukan di file tersebut.

Oleh karena itu, traceability tidak menyatakan bahwa `preprocess.py` melakukan filtering Hentai/Erotica secara langsung.

---

## 3.2 Structured Document

Function:

```text
build_document()
```

mengubah satu baris dataset menjadi structured text yang memuat informasi seperti:

```text
Title
English Title
Type
Episodes
Score
Year
Genres
Themes
Studios
Synopsis
```

Sedangkan `image_url` tidak dimasukkan ke dalam embedding text.

`image_url` dipertahankan sebagai metadata untuk kebutuhan aplikasi/poster.

Function:

```text
build_documents()
```

membangun satu document untuk setiap anime.

Evidence:

```text
15.966 documents
```

disimpan sebagai:

```text
data/processed/anime_documents.jsonl
```

---

# 4. Embedding and Vector Index

## 4.1 Embedding Model

Embedding dilakukan pada:

```text
src/build_index.py
```

Function:

```text
build_index()
```

menggunakan:

```text
sentence-transformers/all-MiniLM-L6-v2
```

dengan:

```text
384 dimensions
```

Embedding juga dinormalisasi.

Validasi dilakukan terhadap dimensi dan L2 norm embedding.

---

## 4.2 FAISS Vector Index

Index dibangun menggunakan:

```text
faiss.IndexFlatIP
```

dengan dimensi sesuai embedding.

Karena embedding dinormalisasi, inner product digunakan untuk semantic similarity pada vector space.

Artifact utama:

```text
anime.index
id_mapping.pkl
```

`id_mapping.pkl` menjaga hubungan:

```text
FAISS vector position ↔ mal_id
```

Source code juga melakukan consistency validation antara:

- jumlah dokumen;
- jumlah vector;
- jumlah mapping;
- dimension index;
- uniqueness `mal_id`;
- positional mapping.

---

# 5. Retrieval

Retrieval diimplementasikan pada:

```text
src/rag_pipeline.py
```

Fungsi utama:

```text
_encode_query()
_search_faiss()
_retrieve_candidates()
retrieve()
retrieve_filtered()
```

Alur utama:

```text
User Query
    ↓
_encode_query()
    ↓
FAISS Search
    ↓
id_mapping
    ↓
Document / Metadata
    ↓
Attribute Filtering
    ↓
Anchor / Franchise Handling
    ↓
Reranking
    ↓
Top-K Results
```

## 5.1 Semantic Retrieval

`_search_faiss()` melakukan vector search terhadap query embedding.

`retrieve_filtered()` kemudian menjadi orchestration layer yang menangani retrieval yang lebih lengkap.

---

## 5.2 Attribute Filtering

Function:

```text
_detect_attribute_filters()
_apply_attribute_filters()
```

mendeteksi dan menerapkan constraint metadata seperti:

- genre;
- theme;
- score;
- year;
- type;
- studio.

Contoh temporal constraint yang telah diverifikasi pada source:

```text
tahun YYYY ke atas
→ min_year = YYYY

setelah/sesudah/lebih dari YYYY
→ min_year = YYYY + 1

tahun YYYY ke bawah
→ max_year = YYYY

sebelum/kurang dari YYYY
→ max_year = YYYY - 1
```

Hasil filtering kemudian digunakan sebelum tahap ranking akhir.

---

# 6. Anchor-Based Retrieval

Untuk query similarity seperti anime yang mirip dengan anime tertentu, sistem dapat menggunakan anime referensi sebagai anchor.

Function:

```text
_find_anchor()
_retrieve_anchor_candidates()
```

`_retrieve_anchor_candidates()` melakukan embedding terhadap document anchor dan menggunakan FAISS untuk mendapatkan kandidat.

Sistem juga memiliki:

```text
_is_same_franchise()
_apply_anchor_exclusion()
```

untuk menangani kandidat dari franchise yang sama.

---

# 7. Reranking

Reranking dilakukan pada:

```text
_rerank()
```

dengan kombinasi:

```text
0.6 × semantic similarity
+
0.4 × normalized MAL score
```

Hasil kemudian diurutkan dan diberi `final_rank`.

Dengan demikian, ranking akhir bukan hanya berdasarkan semantic similarity mentah.

---

# 8. Retrieval Evaluation Evidence

Retrieval telah dievaluasi pada **Phase 9** menggunakan 300 valid test cases:

```text
200 retrieval cases
100 OOS/adversarial cases
```

Untuk 200 retrieval cases:

```text
Errors: 0
```

### Similarity

| Metric |    K=3 |    K=5 |   K=10 |
| ------ | -----: | -----: | -----: |
| P@K    | 0.6867 | 0.6880 | 0.6440 |

### Factual

| Metric   |    K=3 |    K=5 |   K=10 |
| -------- | -----: | -----: | -----: |
| MRR      | 0.9700 |      — |      — |
| Recall@K | 0.9800 | 0.9800 | 0.9800 |

### Attribute

| Metric |    K=3 |    K=5 |   K=10 |
| ------ | -----: | -----: | -----: |
| P@K    | 0.5267 | 0.4840 | 0.4120 |

Primary metric:

```text
P@5 = 0.4840
```

### Multi-turn

| Metric |    K=3 |    K=5 |   K=10 |
| ------ | -----: | -----: | -----: |
| P@K    | 0.2733 | 0.2280 | 0.1700 |

````

Artifact:

```text
results/retrieval_v1.json
````

---

# 9. RAG and Context Augmentation

RAG diimplementasikan pada:

```text
src/rag_pipeline.py
```

Function:

```text
retrieve_filtered()
build_context()
generate()
```

Alur:

```text
Query
  ↓
Retrieval
  ↓
Retrieved Documents
  ↓
build_context()
  ↓
RAG Prompt
  ↓
SLM
  ↓
Response
```

`build_context()` memasukkan informasi retrieved anime seperti:

- title;
- English title;
- type;
- episodes;
- score;
- genres;
- themes;
- year;
- studios;
- synopsis.

`SYSTEM_PROMPT_RAG` menginstruksikan model untuk menggunakan context yang diberikan dan tidak mengarang fakta anime.

---

# 10. Small Language Model

SLM yang digunakan:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Implementasi berada pada:

```text
src/rag_pipeline.py
```

Function:

```text
load_llm()
generate()
```

Model menggunakan konfigurasi 4-bit:

```text
BitsAndBytesConfig
NF4
float16 compute
double quantization
```

Model dimuat dengan:

```text
device_map="auto"
```

Pipeline generation kemudian digunakan oleh `generate()`.

---

# 11. A/B Generation Experiment

Final experiment menggunakan dua kondisi:

### Condition A

```text
SLM-only
```

Tidak menggunakan retrieval/context.

### Condition B

```text
SLM + RAG
```

Menggunakan retrieval dan context.

Final experiment:

```text
300 queries × 2 conditions
= 600 generation calls
```

Artifact:

```text
results/experiment_v1.jsonl
```

Audit experiment menunjukkan:

- 300 query IDs;
- setiap query memiliki A dan B;
- total 600 records;
- tidak ada malformed JSON;
- tidak ada empty answer;
- 16 blocked cases;
- blocked A/B konsisten;
- blocked cases tidak menggunakan retrieval/context.

---

# 12. LLM-as-a-Judge Evidence

Evaluation Phase 12 menggunakan:

```text
results/judge_scores_v22.jsonl
```

Total:

```text
400 records
200 query IDs × A/B
```

Kategori:

```text
similarity
factual
attribute_filtering
multi_turn_refinement
```

Judge:

```text
gemini-3.1-flash-lite
```

Tidak terdapat duplicate `(query_id, condition)` dan seluruh 200 query IDs memiliki pasangan A/B.

---

# 13. Statistical Evaluation Evidence

Phase 13 membandingkan Condition A dan B menggunakan:

```text
Wilcoxon signed-rank test
```

dan rank-biserial effect size.

Artifact:

```text
results/statistical_analysis_v1.json
docs/STATISTICAL_ANALYSIS.md
```

Hasil yang telah dibekukan:

| Metric           |   N | Mean A | Mean B | Median Δ |      W |  p-value | Holm p-value |  Effect |
| ---------------- | --: | -----: | -----: | -------: | -----: | -------: | -----------: | ------: |
| relevance        | 200 |  2.465 |  3.520 |      1.0 | 2044.5 |   < .001 |       < .001 |  0.6190 |
| factual_accuracy | 200 |  3.100 |  3.780 |      0.0 |   2184 |   < .001 |       < .001 |  0.4709 |
| coherence        | 200 |  4.670 |  4.635 |      0.0 |  804.5 | 0.531425 |     0.531425 | -0.0910 |

Interpretasi effect size harus mengikuti kategori yang digunakan oleh source analysis saat ini. Nilai `0.4709` dicatat sebagai effect yang berada pada kategori moderate menurut klasifikasi kode tersebut.

---

# 14. Guardrail

Guardrail terdiri dari beberapa lapisan.

## 14.1 Data Validation

Pada:

```text
src/preprocess.py
```

terdapat validasi terhadap:

```text
hentai
erotica
```

pada metadata genre/theme.

Catatan: source tersebut membuktikan **validation**, bukan proses filtering/deletion secara langsung.

---

## 14.2 System Prompt

Pada:

```text
src/rag_pipeline.py
```

terdapat:

```text
SYSTEM_PROMPT_RAG
SYSTEM_PROMPT_BASELINE
```

yang memberikan instruksi mengenai:

- penggunaan context;
- tidak mengarang fakta;
- konten dewasa;
- out-of-scope;
- perilaku chatbot.

---

## 14.3 Explicit Hard-Block

Pada:

```text
src/guardrails.py
```

fungsi:

```text
guard_query()
```

memeriksa query sebelum diteruskan ke retrieval dan LLM.

Kategori hard-block:

1. explicit/adult content;
2. non-anime topics;
3. personal-profile/history-based recommendation;
4. streaming/download requests.

Function terkait:

```text
is_explicit_request()
is_non_anime_topic()
is_personal_profile_request()
is_streaming_request()
guard_query()
```

---

## 14.4 OOS Diagnostic

Function:

```text
looks_out_of_scope()
```

menggunakan:

```text
OUT_OF_SCOPE_HINTS
```

Namun fungsi ini **bukan hard-block**.

Fungsinya adalah heuristic untuk logging/testing/evaluation.

---

# 15. Multi-turn

Multi-turn diimplementasikan pada:

```text
src/rag_pipeline.py
```

Function:

```text
is_followup()
build_effective_query()
```

Scope implementasinya adalah **2-turn context**.

Query follow-up menggunakan immediate previous user turn untuk membangun effective query.

Secara konseptual:

```text
Turn 1:
"Rekomendasikan anime comedy"

        ↓

Turn 2:
"yang ratingnya di atas 8?"

        ↓

build_effective_query()

        ↓

effective query
```

Catatan:

Sistem bukan conversational memory lintas sesi. Implementasi yang diaudit hanya menggunakan konteks percakapan yang relevan dalam scope dua turn.

---

# 16. Failure Analysis Traceability

Failure analysis Phase 14 digunakan untuk menghubungkan kelemahan sistem dengan komponen pipeline.

Kategori yang digunakan:

```text
Retrieval Failure
Context Utilization Failure
Generation Failure
Constraint Failure
Multi-turn Failure
Guardrail Failure
```

Beberapa pola failure yang telah ditemukan:

### Retrieval Failure

Contoh:

```text
FACT-010
SIM-007
```

menunjukkan kasus ketika informasi target tidak berhasil ditemukan pada retrieval.

### Context Utilization / Generation Failure

Contoh:

```text
SIM-001
SIM-002
FACT-002
```

menunjukkan kasus ketika retrieval dapat menyediakan informasi yang relevan tetapi jawaban akhir masih bermasalah.

### Constraint Failure

Contoh:

```text
ATTR-008
ATTR-017
ATTR-023
ATTR-037
```

menunjukkan masalah ketika constraint metadata tidak sepenuhnya tercermin pada rekomendasi akhir.

### Multi-turn Failure

Contoh:

```text
MULTI-029
MULTI-047
```

menunjukkan keterbatasan dalam mempertahankan kandidat/konteks hasil sebelumnya pada follow-up.

### Guardrail Failure

Contoh:

```text
OOS-002
OOS-012
```

menunjukkan keterbatasan keyword-based hard-block terhadap beberapa pertanyaan out-of-scope.

Dokumen failure analysis lengkap:

```text
docs/FAILURE_ANALYSIS.md
```

---

# 17. Enrichment Status

Jikan/enrichment **tidak digunakan pada final production pipeline**.

Pada `src/rag_pipeline.py`:

```text
enrich()
```

tersedia tetapi implementasinya mengembalikan:

```text
None
```

Dengan demikian:

```text
Jikan enrichment
→ NOT USED in final pipeline
```

Traceability tidak boleh menyatakan bahwa Jikan merupakan komponen final RAG.

Final experiment hanya menggunakan:

```text
Condition A
Condition B
```

dan tidak menggunakan Condition C enrichment.

---

# 18. UI Traceability — Phase 16

Bagian ini **belum difinalkan** karena `app.py` belum diaudit.

Saat Phase 16 selesai, bagian berikut harus dilengkapi:

| Komponen UI            | File     | Function | Evidence yang Harus Diverifikasi   |
| ---------------------- | -------- | -------- | ---------------------------------- |
| Chat interface         | `app.py` | TBD      | Komponen Gradio yang digunakan     |
| User input             | `app.py` | TBD      | Input query → pipeline             |
| Chat history           | `app.py` | TBD      | Cara history diteruskan            |
| Recommendation display | `app.py` | TBD      | Retrieved results yang ditampilkan |
| Poster                 | `app.py` | TBD      | `image_url` → visual poster        |
| Response rendering     | `app.py` | TBD      | Response SLM → UI                  |
| Refusal display        | `app.py` | TBD      | Guardrail refusal → UI             |
| Multi-turn display     | `app.py` | TBD      | Follow-up → chat interface         |
| AniRAG branding        | `app.py` | TBD      | Final UI branding                  |

### Hal yang harus diverifikasi pada Phase 16

Phase 16 harus menjawab secara langsung:

1. Bagaimana query user masuk ke `rag_pipeline.py`.
2. Function apa di `app.py` yang memanggil `generate()`.
3. Bagaimana hasil retrieval ditampilkan.
4. Bagaimana `image_url` digunakan untuk poster.
5. Bagaimana response/refusal ditampilkan.
6. Bagaimana history/chat state diteruskan.
7. Apakah UI benar-benar merepresentasikan alur RAG yang telah diaudit.
8. Apakah tampilan final menggunakan branding **AniRAG-v2** yang telah ditetapkan.
9. Apakah terdapat komponen UI yang tidak sesuai dengan implementasi backend.

**Jangan mengisi bagian ini berdasarkan asumsi sebelum `app.py` diaudit.**

---

# 19. Current Traceability Status

### Sudah terverifikasi

```text
Dataset validation
        ↓
Structured document
        ↓
Embedding
        ↓
Normalization
        ↓
FAISS index
        ↓
ID mapping
        ↓
Semantic retrieval
        ↓
Attribute filtering
        ↓
Anchor retrieval
        ↓
Franchise exclusion
        ↓
Reranking
        ↓
RAG context
        ↓
Llama SLM
        ↓
Guardrail
        ↓
Multi-turn
        ↓
Experiment / Evaluation
```

### Belum terverifikasi

```text
Backend result
      ↓
app.py
      ↓
Gradio UI
      ↓
Chat display
      ↓
Poster
      ↓
User-visible behavior
```

Bagian tersebut akan dilengkapi pada **Phase 16**.

---

# 20. Audit Rules

Dokumen ini mengikuti aturan traceability berikut:

1. Setiap klaim implementasi harus dapat ditunjukkan pada source code.
2. Setiap function yang disebut harus benar-benar terdapat pada file yang dirujuk.
3. Evidence eksperimen harus berasal dari artifact eksperimen yang telah dibekukan.
4. Komponen yang hanya ada pada roadmap atau komentar tidak dianggap sebagai implementasi final.
5. Komponen yang tidak terbukti harus diberi status:
   - `NOT IMPLEMENTED`;
   - `PARTIALLY IMPLEMENTED`; atau
   - `BELUM TERVERIFIKASI`.

6. UI tidak boleh disimpulkan sebelum `app.py` diaudit.
7. Jikan/enrichment tidak dianggap sebagai komponen final karena `enrich()` saat ini tidak melakukan enrichment.
8. `looks_out_of_scope()` tidak dianggap sebagai hard-block.
9. `preprocess.py` hanya dibuktikan melakukan validasi blocked tags; proses filtering/removal harus dibuktikan dari source preprocessing yang memang melakukan operasi tersebut.
10. Final experiment menggunakan A/B, bukan A/B/C.

---

# 21. Phase 16 Completion Checklist

Setelah `app.py` diberikan, update dokumen ini dengan:

- [ ] File `app.py` diaudit.
- [ ] Main UI function ditemukan.
- [ ] Query → `generate()` trace diverifikasi.
- [ ] Retrieval result → UI trace diverifikasi.
- [ ] `image_url` → poster trace diverifikasi.
- [ ] Response → UI trace diverifikasi.
- [ ] Refusal → UI trace diverifikasi.
- [ ] Chat history/state trace diverifikasi.
- [ ] Multi-turn UI behavior diverifikasi.
- [ ] AniRAG-v2 branding diverifikasi.
- [ ] Semua row `UI` pada matrix diperbarui.
- [ ] Status TBD dihapus atau diganti berdasarkan evidence.
- [ ] Final `docs/TRACEABILITY.md` diperbarui setelah audit `app.py`.

---

**Current status: DRAFT / PHASE 15**

Source yang telah digunakan untuk draft ini:

```text
src/preprocess.py
src/build_index.py
src/rag_pipeline.py
src/guardrails.py
```

`app.py` sengaja belum dimasukkan sebagai source yang telah diverifikasi dan akan ditambahkan pada Phase 16.
