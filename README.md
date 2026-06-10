# LLM Construction — POLRAG QA System

A Retrieval-Augmented Generation (RAG) pipeline for question answering over scanned construction/tunnel-repair documents, implementing the **POLRAG** framework.

---

## Overview

The system answers technical questions about tunnel segment repair methods by retrieving relevant passages from OCR-processed PDF pages and generating structured answers via the Gemini LLM.

**Three-stage pipeline:**

1. **Knowledge Base Construction** — OCR JSON files are loaded, chapter-tagged, and chunked into ~300-word passages, then indexed into a FAISS vector store (Gemini, OpenAI, or local-hash embeddings — selectable via `--embeddings`) alongside a BM25 index.
2. **Retrieval** — The user query is optionally rewritten by the LLM into a structured search expression, then recalled via both BM25 keyword search and semantic vector similarity. Results are fused with Reciprocal Rank Fusion (RRF).
3. **Generation** — Retrieved context is injected into a structured POLRAG prompt (system role + context + answer guidelines) and sent to Gemini. An optional second LLM pass cleans noisy OCR content before returning the final answer.

In batch mode the pipeline also **scores each retrieval method automatically** against the ground-truth `Page`/`Answer` columns (page-hit recall/precision, Hit@k, and answer-similarity) and prints a single-vs-fused comparison table.

> **Recommended configuration** (from the evaluation in `report/pilot_findings.md`): `--embeddings gemini --no-query-rewrite`. Real semantic embeddings make the `vector`/`fused` retrievers strong, and query rewriting was found to *hurt* retrieval on this corpus.

---

## Project Structure

```
LLM_Construction/
├── src/
│   └── RAG.py                  # Main RAG pipeline (POLRAG implementation)
├── dataset_json/               # OCR JSON files, one per page, grouped by chapter folder
├── script/
│   ├── QA_generationv2.ipynb   # QA pair generation notebook
│   ├── result_analysis.ipynb   # Evaluation and result analysis notebook
│   ├── PDFtoPNG.py             # Convert PDF pages to PNG images
│   ├── ggOCR.py                # Google Vision OCR runner
│   └── Preprocess_data.ipynb   # Data preprocessing notebook
├── table/                      # Ground-truth QA CSVs for evaluation
├── tablev2/                    # Extended QA CSVs (level-2 difficulty)
├── result/                     # Batch prediction + scored output CSVs
├── report/                     # IEEE paper (conference_101719.tex) + pilot_findings.md
├── .emb_cache/                 # Cached embedding vectors (keyed by text hash)
└── geminikey.txt               # Gemini API keys (one per line, gitignored)
```

> **Key file location:** keys are loaded from `geminikey.txt` **next to `RAG.py` (`src/geminikey.txt`) first**, then the project-root `geminikey.txt`, then the `GOOGLE_API_KEY`/`GEMINI_API_KEY` environment variable.

---

## Requirements

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters \
            faiss-cpu pandas google-genai
```

| Package | Purpose |
|---|---|
| `langchain` / `langchain-community` | Document chunking, FAISS wrapper |
| `faiss-cpu` | Vector similarity index |
| `google-genai` | Gemini LLM generation |
| `langchain-openai` | OpenAI embeddings (optional, falls back to local hash embeddings) |
| `pandas` | Batch CSV evaluation |

---

## API Keys

**Gemini (required):** Create `geminikey.txt` with one API key per line — in `src/` (checked first) or the project root. Multiple keys are supported; the system rotates through them automatically on rate-limit errors. Used for generation, query rewriting, and (optionally) embeddings.

```
GEMINI_API_KEY_1
GEMINI_API_KEY_2
```

Alternatively, set the `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable. Verify connectivity at any time with `python src/RAG.py --docs dataset_json --check-gemini` (pings the API and exits without building the knowledge base).

**Embeddings backend** is chosen with `--embeddings`:

| `--embeddings` | Embedder | Notes |
|---|---|---|
| `gemini` | `gemini-embedding-001` (3072-dim) | **Recommended.** Real multilingual semantics; reuses the Gemini keys; vectors cached in `.emb_cache/`. |
| `openai` | OpenAI embeddings | Requires `OPENAI_API_KEY`. |
| `hash` | `LocalHashEmbeddings` (512-dim) | Deterministic placeholder, **no semantics** — for offline smoke tests only. |
| `auto` *(default)* | OpenAI if `OPENAI_API_KEY` is set, else `hash` | Backward-compatible default. |

---

## Usage

### Single query
```bash
python src/RAG.py --docs dataset_json \
  --query "เกณฑ์การซ่อมแซมรอยร้าว (Crack Repair) สำหรับรอยร้าวแบบ Hard Crack คืออะไร"
```

### Interactive multi-turn conversation
```bash
python src/RAG.py --docs dataset_json --interactive
```
Type `reset` to clear conversation history, `quit` to exit.

### Batch CSV evaluation (recommended config)
```bash
python src/RAG.py --docs dataset_json \
  --qa-csv tablev2/QA_test.csv \
  --output-csv result/QA_test_out.csv \
  --embeddings gemini --no-query-rewrite --modes keyword,vector,fused
```
Reads questions from the CSV, runs the selected retrieval modes, writes predictions **and per-method scores** (`PageRecall`, `PagePrec`, `Hit`, `AnsSim`) back to the output CSV, and prints a single-vs-fused comparison table. Use `--limit N` to run only the first N questions (handy for a quick pilot).

### Retrieval-only sweep (no LLM, free)
```bash
python src/RAG.py --docs dataset_json --qa-csv tablev2/QA_test.csv \
  --retrieval-only --no-query-rewrite --embeddings gemini
```
Scores page-hit per retrieval method **without any Gemini generation** — fast and quota-free for tuning `--rrf-k`, `--top-k-each/-fused`, and `--bm25-k1/-b`.

### Show retrieved context
Add `--show-context` to any mode to print the retrieved passages before the answer.

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--docs` | *(required)* | Directory containing source JSON files |
| `--query` | — | Single question (single-query mode) |
| `--interactive` | `false` | Enable multi-turn conversation mode |
| `--qa-csv` | — | Input CSV for batch evaluation |
| `--output-csv` | — | Output CSV path (defaults to `<input>_predicted.csv`) |
| `--question-col` | `Question` | Column name for questions in `--qa-csv` |
| `--gemini-model` | `gemini-2.5-flash` | Gemini model ID |
| `--embeddings` | `auto` | Embedding backend: `gemini`, `openai`, `hash`, or `auto` |
| `--chunk-size` | `1500` | Chunk size in characters (~300 words) |
| `--chunk-overlap` | `200` | Overlap between chunks |
| `--top-k-each` | `8` | Top-K candidates per retriever before fusion |
| `--top-k-fused` | `5` | Final top-K chunks passed to the LLM |
| `--rrf-k` | `60` | RRF constant (Robertson et al. 2009) |
| `--bm25-k1` | `1.5` | BM25 term-frequency saturation |
| `--bm25-b` | `0.75` | BM25 length-normalisation strength |
| `--embedding-dim` | `512` | Dimension for `--embeddings hash` only |
| `--modes` | `keyword,vector,fused` | Comma list of retrieval modes to run (`keyword`,`fulltext`,`vector`,`fused`) |
| `--limit` | — | Run only the first N questions of `--qa-csv` |
| `--no-query-rewrite` | `false` | Skip LLM query rewriting step |
| `--no-reformat` | `false` | Skip the second LLM answer-reformatting pass |
| `--retrieval-only` | `false` | Score page-hit per mode with no LLM generation |
| `--check-gemini` | `false` | Ping the Gemini API and exit (no KB build) |
| `--show-context` | `false` | Print retrieved context before the answer |

---

## Data Format

Each chapter is a subfolder under `dataset_json/`. Each page is a JSON file named `page_<N>.json`. The JSON is a list of OCR token objects; the first element's `description` field contains the full page text.

```
dataset_json/
└── 3_Repair_With_Epoxy_Resin/
    ├── page_1.json
    ├── page_2.json
    └── ...
```

Folder names follow the pattern `<chapter_number>_<Chapter_Title_Words>`. The chapter title is parsed automatically and attached to each chunk as metadata for citation in answers.

---

## Output Format

Every answer includes a page reference derived from the retrieved chunk metadata:

```
Page:   3, 5
Answer: 1. Clean the crack surface...
```

In batch CSV mode, one set of columns is written **per selected retrieval mode** (`--modes`), including automatic per-method scores:

```
predicted_fused_Answer
predicted_fused_Page
predicted_fused_Category
predicted_fused_Rainbow Group
predicted_fused_answer_model
predicted_fused_PageRecall      # GT pages retrieved / GT pages
predicted_fused_PagePrec        # correct pages / retrieved pages
predicted_fused_Hit             # 1 if any GT page was retrieved
predicted_fused_AnsSim          # cosine similarity of answer vs ground truth
...
compute_time_s                  # per-question wall time
```

A per-method comparison table (single methods vs. fused, on Recall/Prec/Hit@k/AnsSim) is also printed to the console at the end of the run.

---

## Retrieval Methods

### 1. BM25 (Best Match 25) — Keyword Retrieval

BM25 is a classic term-frequency ranking function. It scores every document chunk against the query by counting shared keywords, while penalising very common words (low IDF) and very long documents (length normalisation).

**Score formula for a query Q against document D:**

```
         |Q|
BM25 =   Σ   IDF(qi) × TF_norm(qi, D)
         i=1

         tf(qi, D) × (k1 + 1)
TF_norm = ─────────────────────────────────────
          tf(qi, D) + k1 × (1 - b + b × |D|/avgdl)

IDF(qi) = log( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )
```

| Symbol | Meaning | Value used |
|---|---|---|
| `tf(qi, D)` | How many times query term `qi` appears in document `D` | counted at runtime |
| `\|D\|` | Length of document `D` in tokens | counted at runtime |
| `avgdl` | Average document length across all chunks | computed at index time |
| `N` | Total number of chunks | computed at index time |
| `df(qi)` | Number of chunks containing term `qi` | computed at index time |
| `k1` | Term-frequency saturation — higher = more weight to repeated terms | **1.5** |
| `b` | Length normalisation strength — 1.0 = full, 0.0 = none | **0.75** |

**Worked example:**

Suppose the query is `"crack repair"` and one chunk contains the word `"crack"` 3 times out of 200 tokens, with `avgdl = 150` and `IDF("crack") = 2.1`:

```
TF_norm = 3 × (1.5 + 1) / (3 + 1.5 × (1 - 0.75 + 0.75 × 200/150))
        = 7.5 / (3 + 1.5 × 1.25)
        = 7.5 / 4.875
        ≈ 1.538

BM25 contribution of "crack" = 2.1 × 1.538 ≈ 3.23
```

All query terms are summed. Chunks are ranked by total BM25 score descending.

---

### 2. Vector Similarity — Semantic Retrieval

Each chunk and the query are converted to dense embedding vectors. Retrieval finds the chunks whose vectors are closest to the query vector in the embedding space, capturing semantic meaning rather than exact word matches.

**Similarity measure:** cosine similarity (via FAISS `IndexFlatL2` internally — equivalent to cosine on normalised vectors).

```
             A · B
cos(A, B) = ───────
            ‖A‖ ‖B‖
```

A value of `1.0` means identical direction (most similar); `0.0` means orthogonal.

**Embedding options (selected via `--embeddings`):**

| `--embeddings` | Embedder used | Dimension |
|---|---|---|
| `gemini` | `gemini-embedding-001` (Gemini) | 3072 |
| `openai` | OpenAI embeddings (requires `OPENAI_API_KEY`) | 1536 |
| `hash` | `LocalHashEmbeddings` (SHA-256 hash) | 512 |
| `auto` | OpenAI if keyed, else hash | — |

> With the non-semantic `hash` embedder the `vector` retriever is effectively random; use `gemini` (or `openai`) for meaningful semantic retrieval. Gemini vectors are cached on disk in `.emb_cache/` so chunks are embedded only once.

**Example:** The query `"วิธีซ่อมรอยร้าว"` (how to repair cracks) will score highly against a chunk discussing `"crack repair procedures"` even if no exact Thai words appear in the chunk, because the embedding model maps them to nearby vectors.

---

### 3. Reciprocal Rank Fusion (RRF) — Score Fusion

RRF combines the BM25 ranking and the vector ranking into a single ranked list without needing to normalise or compare their raw scores. Each chunk receives a score based on its position in each ranking.

**Formula:**

```
           Σ         1
RRF(d) =  ────────────────
         rankings  k + rank(d)
```

| Symbol | Meaning | Value used |
|---|---|---|
| `rank(d)` | Position of chunk `d` in a given ranking (1 = top) | from BM25 or vector |
| `k` | Smoothing constant — reduces the dominance of rank-1 | **60** |

RRF scores from all rankings are summed per chunk. Chunks are then re-ranked by total RRF score descending, and the top `top_k_fused` (default **5**) are kept.

**Worked example with k = 60:**

| Chunk | BM25 rank | Vector rank | RRF score |
|---|---|---|---|
| A | 1 | 3 | 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = **0.03226** |
| B | 3 | 1 | 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = **0.03226** |
| C | 2 | 5 | 1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = **0.03151** |

Chunks A and B tie because they both appear in top-3 of both methods. Chunk C ranks third.

**Why RRF?** It is robust to score scale differences between BM25 (unbounded floats) and cosine similarity (0–1). A chunk that ranks well in both methods reliably floats to the top.

---

### 4. Query Rewriting

Before retrieval, the raw user question is rewritten by Gemini into a structured search expression. This bridges the gap between conversational phrasing and document vocabulary.

**Prompt template:**
```
Convert the following user question into a structured search expression
for technical document retrieval about tunnel segment repair.
Output format: subject: <subject>; elements: <element1>, <element2>, ...

Q: {user_query}
A:
```

**Example:**

| Input | Rewritten |
|---|---|
| `"วิธีซ่อมโพรงอากาศคืออะไร"` | `subject: void and blowhole repair; elements: repair method, materials, steps` |
| `"How do I fix cracks using epoxy?"` | `subject: crack repair; elements: epoxy resin, procedure, application steps` |

If the LLM call fails for any reason, the original query is used as a fallback (`enable_query_rewrite=False` skips this step entirely via `--no-query-rewrite`).

---

## `src/RAG.py` — Module Reference

### Classes

#### `LocalHashEmbeddings`

A dependency-free fallback embedder used when no `OPENAI_API_KEY` is set. Converts text into a deterministic 512-dimensional vector by hashing each token with SHA-256 and accumulating sign-weighted counts, then L2-normalising the result. Supports both Thai Unicode and ASCII/numeric tokens.

| Method | Description |
|---|---|
| `embed_documents(texts)` | Embed a list of strings → `list[list[float]]` |
| `embed_query(text)` | Embed a single query string → `list[float]` |

---

#### `GeminiEmbeddings`

Real semantic embeddings via the Gemini embedding API (`gemini-embedding-001`, 3072-dim). Reuses the project's Gemini keys (rotating on rate limits), batches requests, and **caches each text's vector by SHA-256** to `.emb_cache/<model>.json` so repeated runs do not re-embed unchanged chunks. Same `embed_documents` / `embed_query` interface as above. Selected with `--embeddings gemini`.

---

#### `RAGFlow`

The main pipeline class. Instantiating it builds the full knowledge base (loads JSONs, chunks, indexes FAISS + BM25). All subsequent calls are stateless except for `conversation_history`.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `documents_dir` | `str` | *(required)* | Path to the OCR JSON folder |
| `chunk_size` | `int` | `1500` | Characters per chunk (~300 words) |
| `chunk_overlap` | `int` | `200` | Overlap between adjacent chunks |
| `gemini_model` | `str` | `"gemini-2.5-flash"` | Primary Gemini model ID |
| `gemini_api_key` | `str \| None` | `None` | Override key (else reads `geminikey.txt`) |
| `top_k_each` | `int` | `8` | Candidates returned by each retriever |
| `top_k_fused` | `int` | `5` | Chunks kept after RRF fusion |
| `rrf_k` | `int` | `60` | RRF smoothing constant |
| `enable_query_rewrite` | `bool` | `True` | Run LLM query-rewriting step |
| `bm25_k1` | `float` | `1.5` | BM25 term-frequency saturation |
| `bm25_b` | `float` | `0.75` | BM25 length-normalisation strength |
| `enable_reformat` | `bool` | `True` | Run the second answer-reformatting LLM pass |
| `embedding_dim` | `int` | `512` | Dimension for the hash embedder |
| `modes` | `list[str]` | `["keyword","vector","fused"]` | Retrieval modes to run in batch mode |
| `embeddings_backend` | `str` | `"auto"` | `gemini` / `openai` / `hash` / `auto` |
| `gemini_embed_model` | `str` | `"gemini-embedding-001"` | Gemini embedding model ID |
| `require_gemini` | `bool` | `True` | If `False`, allow construction without genai/keys (retrieval-only) |

---

### Key Methods

#### `ask(query: str) → dict`

Single-turn or multi-turn question answering. Appends the exchange to `conversation_history` so subsequent calls have context.

**Input:** a natural-language question string (Thai or English).

**Output dict:**

| Key | Type | Description |
|---|---|---|
| `query` | `str` | Original question |
| `rewritten_query` | `str` | LLM-rewritten retrieval expression |
| `retrieval.bm25_rank` | `list[int]` | Document indices ranked by BM25 |
| `retrieval.vector_rank` | `list[int]` | Document indices ranked by FAISS |
| `retrieval.fused_rank` | `list[int]` | Final fused document indices |
| `context` | `str` | Formatted retrieved passages with chapter/page citations |
| `answer_model` | `str` | Gemini model ID that produced the answer |
| `page` | `str` | Comma-separated page numbers from chunk metadata |
| `answer` | `str` | Clean, reformatted answer |

---

#### `predict_expected_columns(question: str) → dict`

Batch evaluation mode. Runs the selected retrieval strategies (`self.modes`) in one call and returns predictions for each.

**Input:** a question string.

**Output dict:**

| Key | Type | Description |
|---|---|---|
| `question` | `str` | Original question |
| `rewritten_query` | `str` | LLM-rewritten retrieval expression |
| `retrieval` | `dict` | Same BM25/vector/fused rank lists as `ask()` |
| `by_mode` | `dict` | Per-mode results, one key per entry in `self.modes` (subset of `"keyword"`, `"fulltext"`, `"vector"`, `"fused"`) |

Each `by_mode[mode]` entry contains:

| Key | Type | Description |
|---|---|---|
| `answer_model` | `str` | Gemini model used |
| `raw_answer` | `str` | Unprocessed LLM output |
| `predictions` | `dict` | Normalised fields: `predicted_Rainbow Group`, `predicted_Category`, `predicted_Answer`, `predicted_Page` |

---

#### `reset_conversation() → None`

Clears the multi-turn `conversation_history` list.

---

### Internal Pipeline Steps

```
ask(query)
  │
  ├─ 1. _rewrite_query()          LLM rewrites query → structured search expression
  │
  ├─ 2. _bm25_retrieval()         Score all chunks with BM25 (k1=1.5, b=0.75)
  ├─ 2. _vector_retrieval()       FAISS cosine similarity search
  │
  ├─ 3. _rrf_fusion()             Combine BM25 + vector rankings via RRF
  │
  ├─ 4. _build_context()          Format top-K chunks with chapter + page citations
  ├─ 4. _build_polrag_prompt()    Assemble: system role + context + history + guidelines
  │
  ├─ 5. _generate_with_gemini()   Call Gemini; auto-rotate API key on rate-limit
  │
  ├─ 6. _parse_response()         Extract Page / Answer fields from LLM output
  └─ 7. _reformat_answer()        Optional second LLM pass to strip OCR noise (disable with --no-reformat)
```

**Retrieval modes used in batch evaluation** (default: `keyword,vector,fused`):

| Mode | Retrieval source |
|---|---|
| `keyword` | BM25 top-K |
| `fulltext` | BM25 top-K (term-based full-text alias — identical to `keyword`; omitted by default) |
| `vector` | FAISS top-K |
| `fused` | RRF fusion of BM25 + FAISS |

### Module-level helpers

| Function | Description |
|---|---|
| `load_api_keys(key, verbose)` | Resolve Gemini keys (`src/geminikey.txt` → root → env var) |
| `ping_gemini(keys, model)` | One-shot API reachability check; returns `(ok, message)` |
| `summarize_modes(frame, modes)` | Mean page-hit / answer-similarity per mode from a scored results frame |
| `format_mode_table(summary, include_sim)` | Render the per-method comparison as a Markdown table |

---

## Reference

This implementation is based on the POLRAG framework described in:

> H. Lin, P. Deng, Q. Zhong, and X. Zhu, "POLRAG: A RAG-LLM Framework for Policy Question Answering," in *Proc. 2025 IEEE Int. Conf. High Performance Computing and Communications (HPCC)*, 2025, pp. 1259–1264, doi: 10.1109/HPCC67675.2025.00179.

An evaluation of this implementation on the tunnel-segment-repair corpus — including the per-method retrieval comparison and the rubric scores — is written up in `report/` (`conference_101719.tex` and `pilot_findings.md`).
