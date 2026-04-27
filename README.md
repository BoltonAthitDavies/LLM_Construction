# LLM Construction — POLRAG QA System

A Retrieval-Augmented Generation (RAG) pipeline for question answering over scanned construction/tunnel-repair documents, implementing the **POLRAG** framework.

---

## Overview

The system answers technical questions about tunnel segment repair methods by retrieving relevant passages from OCR-processed PDF pages and generating structured answers via the Gemini LLM.

**Three-stage pipeline:**

1. **Knowledge Base Construction** — OCR JSON files are loaded, chapter-tagged, and chunked into ~300-word passages, then indexed into a FAISS vector store alongside a BM25 index.
2. **Retrieval** — The user query is rewritten by the LLM into a structured search expression, then recalled via both BM25 keyword search and semantic vector similarity. Results are fused with Reciprocal Rank Fusion (RRF).
3. **Generation** — Retrieved context is injected into a structured POLRAG prompt (system role + context + answer guidelines) and sent to Gemini. A second LLM pass cleans noisy OCR content before returning the final answer.

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
└── geminikey.txt               # Gemini API keys (one per line, gitignored)
```

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

**Gemini (required):** Create `geminikey.txt` in the project root with one API key per line. Multiple keys are supported — the system rotates through them automatically on rate-limit errors.

```
GEMINI_API_KEY_1
GEMINI_API_KEY_2
```

Alternatively, set the `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable.

**OpenAI (optional):** Set `OPENAI_API_KEY` to use `text-embedding-ada-002` for semantic retrieval. Without it, a deterministic local hash embedding is used as a fallback.

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

### Batch CSV evaluation
```bash
python src/RAG.py --docs dataset_json \
  --qa-csv tablev2/QA_test_lvl2.csv \
  --output-csv QA_test_lvl2_out.csv
```
Reads questions from the CSV, runs all four retrieval modes (`keyword`, `fulltext`, `vector`, `fused`), and writes predictions back to the output CSV.

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
| `--chunk-size` | `1500` | Chunk size in characters (~300 words) |
| `--chunk-overlap` | `200` | Overlap between chunks |
| `--top-k-each` | `8` | Top-K candidates per retriever before fusion |
| `--top-k-fused` | `5` | Final top-K chunks passed to the LLM |
| `--rrf-k` | `60` | RRF constant (Robertson et al. 2009) |
| `--no-query-rewrite` | `false` | Skip LLM query rewriting step |
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

In batch CSV mode, four sets of columns are written — one per retrieval mode:

```
predicted_fused_Answer
predicted_fused_Page
predicted_fused_Category
predicted_fused_Rainbow Group
predicted_fused_answer_model
...
```

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

Batch evaluation mode. Runs all four retrieval strategies in one call and returns predictions for each.

**Input:** a question string.

**Output dict:**

| Key | Type | Description |
|---|---|---|
| `question` | `str` | Original question |
| `rewritten_query` | `str` | LLM-rewritten retrieval expression |
| `retrieval` | `dict` | Same BM25/vector/fused rank lists as `ask()` |
| `by_mode` | `dict` | Per-mode results keyed by `"keyword"`, `"fulltext"`, `"vector"`, `"fused"` |

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
  └─ 7. _reformat_answer()        Second LLM pass to strip OCR noise and reformat
```

**Retrieval modes used in batch evaluation:**

| Mode | Retrieval source |
|---|---|
| `keyword` | BM25 top-K |
| `fulltext` | BM25 top-K (term-based full-text alias) |
| `vector` | FAISS top-K |
| `fused` | RRF fusion of BM25 + FAISS |

---

## Reference

This implementation is based on the POLRAG framework described in:

> *POLRAG: A RAG-LLM Framework for Policy Question Answering*
