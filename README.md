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

## Reference

This implementation is based on the POLRAG framework described in:

> *POLRAG: A RAG-LLM Framework for Policy Question Answering*
