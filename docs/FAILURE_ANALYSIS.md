# Phase 14 — Failure Analysis

## 1. Tujuan

Failure analysis dilakukan untuk mengidentifikasi sumber kegagalan
AniRAG-v2 berdasarkan hasil retrieval evaluation, eksperimen Condition A/B,
dan LLM-as-a-Judge.

Analisis menggunakan enam kategori:

1. Retrieval Failure
2. Context Utilization Failure
3. Generation Failure
4. Constraint Failure
5. Multi-turn Failure
6. Guardrail Failure

Setiap kasus diklasifikasikan berdasarkan sumber kegagalan utama.

---

## 2. Retrieval Failure

### F14-01 — FACT-010

Query:
"Golgo 13 termasuk genre apa?"

Ground truth:
MAL ID 1760.

Retrieved:
MAL ID 1760 tidak ditemukan pada top-10.

Recall@3:
0.0

Recall@5:
0.0

Recall@10:
0.0

Retrieval mode:
user_query

Failure:
Retriever gagal menemukan dokumen anime yang sebenarnya tersedia
dalam knowledge base.

Interpretasi:
Kegagalan terjadi pada tahap retrieval sebelum informasi diberikan
kepada SLM. Karena evidence yang relevan tidak masuk ke context,
generator tidak memiliki sumber KB yang tepat untuk menjawab pertanyaan.

Possible cause:
Semantic retrieval berbasis embedding tidak menempatkan dokumen
target dalam kandidat top-K.

---

## 3. Context Utilization Failure

### F14-02 — SIM-001

Query:
"Aku suka Ookami Shoujo to Kuro Ouji: Gishinanki – Happening Kiss,
ada rekomendasi anime lain yang mirip?"

Condition B:
Retrieval menggunakan anchor_document.

Retrieval:
10 dokumen berhasil diperoleh.

P@5:
1.0

Failure:
Model menyatakan tidak menemukan informasi mengenai anime tersebut,
meskipun evidence retrieval tersedia.

Interpretasi:
Retrieval berhasil, tetapi informasi pada context tidak dimanfaatkan
secara tepat oleh generator.

---

### F14-03 — SIM-002

Query:
"Aku suka Yofukashi no Uta Season 2, ada rekomendasi anime lain
yang mirip?"

Condition B:
Retrieval menggunakan anchor_document.

P@5:
0.8

Failure:
Model menyatakan tidak menemukan informasi mengenai anime yang diminta,
padahal retrieval menghasilkan dokumen terkait yang dapat digunakan
sebagai anchor.

Interpretasi:
Kasus menunjukkan kegagalan pemanfaatan context setelah retrieval,
bukan kegagalan retrieval murni.

---

## 4. Generation Failure

### F14-04 — FACT-002 / Condition A

Query:
"Ninja to Koroshiya no Futarigurashi diproduksi oleh studio apa?"

Condition A:
SLM-only.

Answer:
Model menjawab bahwa studio tersebut adalah Madhouse.

Condition B:
Retrieval menemukan:
Ninja to Koroshiya no Futarigurashi
Studio: Shaft

Failure:
Condition A menghasilkan klaim faktual yang tidak didukung oleh
knowledge base.

Interpretasi:
Tanpa grounding dari retrieval, SLM menghasilkan informasi faktual
yang salah.

---

## 5. Constraint Failure

### F14-05 — MULTI-047

Query:
"Dari rekomendasi tadi, pilih yang memenuhi batas tersebut:
yang skornya minimal 8."

Retrieval evidence:
Tsurune: Tsunagari no Issha
Score: 8.11

Failure:
Jawaban menambahkan anime lain yang tidak terdapat dalam context dan
memberikan score yang tidak sesuai dengan KB, termasuk anime yang
seharusnya tidak memenuhi constraint.

Interpretasi:
Retrieval telah menerapkan filter score >= 8, tetapi generator tidak
mempertahankan batasan tersebut ketika menyusun jawaban akhir.

---

## 6. Multi-turn Failure

### F14-06 — MULTI-029

Turn 1:
"Rekomendasikan anime comedy"

Turn 2:
"Dari hasil sebelumnya, ambil yang memenuhi filter itu:
pilih yang tayang tahun 2020 ke atas"

Expected:
Menyaring hasil dari turn sebelumnya.

Actual retrieval:
Retrieval melakukan pencarian berdasarkan query follow-up dan
menghasilkan anime yang memenuhi tahun >= 2020 secara umum.

Failure:
Himpunan kandidat dari turn sebelumnya tidak dipertahankan sebagai
basis filtering.

Interpretasi:
Pipeline belum sepenuhnya melakukan filtering terhadap hasil
rekomendasi sebelumnya pada follow-up multi-turn.

---

## 7. Guardrail Failure

### F14-07 — OOS-002

Query:
"Bagaimana cara membuat REST API dengan Java?"

Classification:
Out-of-scope.

Actual:
Query tidak diblokir dan chatbot menghasilkan penjelasan mengenai
REST API Java.

Failure:
Guardrail tidak mengenali permintaan non-anime tersebut sebagai
out-of-scope.

---

### F14-08 — OOS-012

Query:
"Hitung 125 dikali 37."

Classification:
Out-of-scope.

Condition A:
Query tidak diblokir dan menghasilkan jawaban perhitungan.

Condition B:
Query tidak diblokir dan bahkan memicu retrieval anime.

Failure:
Permintaan non-anime melewati guardrail dan pada Condition B masuk
ke pipeline retrieval.

---

## 8. Ringkasan Failure Analysis

| Category                    | Representative Cases | Main Failure                                                    |
| --------------------------- | -------------------- | --------------------------------------------------------------- |
| Retrieval Failure           | FACT-010             | Dokumen relevan tidak ditemukan                                 |
| Context Utilization Failure | SIM-001, SIM-002     | Evidence tersedia tetapi tidak dimanfaatkan                     |
| Generation Failure          | FACT-002/A           | SLM menghasilkan fakta yang salah                               |
| Constraint Failure          | MULTI-047            | Constraint retrieval tidak dipertahankan pada jawaban           |
| Multi-turn Failure          | MULTI-029            | Hasil turn sebelumnya tidak dipertahankan sebagai candidate set |
| Guardrail Failure           | OOS-002, OOS-012     | Query non-anime melewati guardrail                              |

## 9. Kesimpulan

Failure analysis menunjukkan bahwa kegagalan AniRAG-v2 tidak berasal dari
satu komponen saja. Kegagalan dapat terjadi pada tahap retrieval,
pemanfaatan context, generation, pemenuhan constraint, pemrosesan
multi-turn, maupun guardrail.

Temuan ini digunakan sebagai dasar untuk pembahasan keterbatasan sistem
dan tidak digunakan sebagai alasan untuk menjalankan ulang eksperimen
utama.
