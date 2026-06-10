# Pilot Findings — SmartCivil RAG Experiment (Step 3)

**Date:** 2026-06-09
**Scope:** Check Gemini availability, then run the experiment on 5 Level-1 questions
(`tablev2/QA_test.csv`) to see the big picture before committing to the full run.

---

## 0. Environment status

This machine was **missing two required packages**; both were installed into the interpreter
that runs `RAG.py` (base miniconda) via `python -m pip`:

| Package | Status before | Now |
|---|---|---|
| `faiss-cpu` (vector store) | not installed | 1.14.2 |
| `google-genai` (LLM client) | not installed | 2.8.0 |

> Reminder: `python` here is base miniconda; the `pip` on PATH points at the `qa` conda env.
> Always use **`python -m pip install ...`** so deps land in the right interpreter.

## 1. Gemini availability — OK ✅ (after key update)

- The **old keys in the project-root `geminikey.txt` are leaked/revoked** (all 10 →
  `403 PERMISSION_DENIED, "API key was reported as leaked"`).
- The **new keys you put in `src/geminikey.txt` work**: tested individually, **5 of 6 are valid**;
  key index 5 returns `400 INVALID_ARGUMENT` (malformed/truncated — worth re-pasting).
- **Loader fix:** `load_api_keys` now reads `geminikey.txt` **next to `RAG.py` first**, then falls
  back to the project root. So `src/geminikey.txt` is now the active key file.
  `--check-gemini` → `OK — reachable (gemini-2.5-flash); sample reply: 'Pong!'`

**Security:** the root `geminikey.txt` still contains leaked keys committed in the repo. Delete it
(or scrub it), keep key files out of version control (`.gitignore`), and rotate anything exposed.

## 2. Full generation pilot — 5 Level-1 questions

```
python src/RAG.py --docs dataset_json --qa-csv tablev2/QA_test.csv \
  --output-csv result/result_pilot_lvl1.csv --limit 5 --no-reformat --modes keyword,vector,fused
```
5 questions in **116.3 s — avg 23.3 s/question** (reformat pass disabled). Scoring is automatic:
page-hit vs the GT `Page` column, plus answer-similarity (`AnsSim`, cosine of answer embeddings).

### Single methods vs. combined (fused) — with query rewriting + generation

| Mode | PageRecall | PagePrec | Hit@k | AnsSim |
|---|---|---|---|---|
| keyword (single) | 0.200 | 0.040 | 0.200 | 0.205 |
| vector (single)  | 0.000 | 0.000 | 0.000 | 0.299 |
| fused (combined) | 0.200 | 0.040 | 0.200 | 0.190 |

### Retrieval-only baseline (same 5 Qs, **no query rewrite, no LLM**)

| Mode | PageRecall | PagePrec | Hit@k |
|---|---|---|---|
| keyword (single)  | 0.400 | 0.090 | 0.400 |
| fulltext (single) | 0.400 | 0.090 | 0.400 |
| vector (single)   | 0.200 | 0.040 | 0.200 |
| fused (combined)  | 0.400 | 0.080 | 0.400 |

## 3. Observations

1. **Query rewriting HURT retrieval here.** Page-hit halved when the LLM query-rewrite was on
   (keyword/fused 0.40 → 0.20, vector 0.20 → 0.00). The rewritten "subject/elements" expression
   seems to drift from the document wording. **Action: A/B test `--no-query-rewrite`** — it may be
   a net win, and it removes one LLM call per question.
2. **`vector` mode is still crippled.** No `OPENAI_API_KEY`, so dense retrieval uses
   `LocalHashEmbeddings` (a SHA-256 bag-of-tokens hash, not semantics) — hence vector recall 0.00.
   `AnsSim` uses the same weak embedding, so its numbers are only a rough relative signal (note
   vector's higher AnsSim despite 0 page-hit is an artefact, not real quality).
3. **`keyword` == `fulltext`** (both BM25) — identical, as expected; not yet two real methods.
4. **Fused ≈ BM25**: with vector dead, RRF fusion adds nothing. Fusion should only help once dense
   retrieval is competitive.
5. **Latency ~23 s/question** (still > 20 s even with reformat off) → would score 0/10 on the
   latency rubric. Dropping query-rewrite and the reformat pass are the obvious levers.
6. **Retrieval is the bottleneck**, not generation: when the right page is retrieved the answers
   are accurate, but the right page is found for only ~1–2 of 5 questions.

## 4. Recommended next steps (before the full 20-question run)

1. **Re-paste key index 5** in `src/geminikey.txt` (currently 400 INVALID_ARGUMENT); delete the
   leaked root `geminikey.txt` and gitignore key files.
2. **Fix dense retrieval first** — set `OPENAI_API_KEY` or wire a real multilingual/Thai embedding
   model, then re-run the retrieval-only pilot to confirm vector ≥ BM25.
3. **A/B query rewriting** with `--retrieval-only` (free): compare default vs `--no-query-rewrite`
   on all 20 questions; keep whichever maximises `Hit@k`/`Recall`.
4. **Sweep retrieval hyperparameters cheaply** via `--retrieval-only`: `--rrf-k`,
   `--top-k-each/-fused`, `--bm25-k1/-b`.
5. **Then run the full generation experiment** (both `QA_test.csv` and `QA_test_lvl2.csv`) and use
   the per-method comparison table for the paper's results section.

## 5. Query-rewrite A/B on the FULL QA sets (all 40 questions, retrieval-only)

Ran every question in both files, rewrite ON vs OFF, no generation (rewrite-on uses one cheap
LLM rewrite call per question; rewrite-off is free). Modes: keyword (=fulltext), vector, fused.

**Level 1 (`QA_test.csv`, 20 Q)**

| Mode | Recall OFF → ON | Hit@k OFF → ON |
|---|---|---|
| keyword | 0.450 → 0.400 | 0.450 → 0.400 |
| vector  | 0.250 → 0.250 | 0.250 → 0.250 |
| fused   | 0.400 → 0.400 | 0.400 → 0.400 |

**Level 2 (`QA_test_lvl2.csv`, 20 Q)**

| Mode | Recall OFF → ON | Hit@k OFF → ON |
|---|---|---|
| keyword | 0.567 → 0.417 | **0.750 → 0.550** |
| vector  | 0.300 → 0.375 | 0.400 → 0.500 |
| fused   | 0.517 → 0.392 | 0.650 → 0.550 |

### Verdict
- **Query rewriting hurts the dominant keyword/BM25 path and fused**, and the damage is large on
  the harder multi-category Level-2 set (keyword Hit@k **0.75 → 0.55**). It only helps the weak
  `vector` path (which is hash-embedding noise anyway).
- **Recommendation: run with `--no-query-rewrite`.** It gives the best retrieval (keyword OFF:
  L1 Hit 0.45, L2 Hit 0.75) *and* removes one LLM call + latency per question.
- Best current configuration: **BM25 keyword retrieval, rewrite off** — it beats fused everywhere
  here, because dense retrieval is still crippled (no real embeddings) so RRF has nothing to add.

CSV evidence: `result/sweep_lvl1_norewrite.csv`, `sweep_lvl1_rewrite.csv`,
`sweep_lvl2_norewrite.csv`, `sweep_lvl2_rewrite.csv`.

### Still blocked on you
- **Vector retrieval**: needs `OPENAI_API_KEY` (or a real Thai/multilingual embedding model) to
  become meaningful — until then keyword wins by default and fusion is pointless.
- **Key #5** in `src/geminikey.txt` is malformed (400) — re-paste it.

## 6. Real embeddings wired in — Gemini `gemini-embedding-001` ✅

Added a `GeminiEmbeddings` adapter (3072-dim, reuses `src/geminikey.txt`, key-rotating) with an
**on-disk cache** (`.emb_cache/`) so the 218 chunks are embedded once, not every run. Selected via
`--embeddings gemini`. This fixes the dead `vector` path that crippled every earlier result.

**Full QA, retrieval-only, `--no-query-rewrite` — hash vs. real Gemini embeddings (Hit@k):**

| | keyword | vector (hash → gemini) | fused (hash → gemini) |
|---|---|---|---|
| **Level 1** (20 Q) | 0.450 | 0.250 → **0.650** | 0.400 → **0.750** |
| **Level 2** (20 Q) | 0.750 | 0.400 → **0.950** | 0.650 → **0.850** |

### Verdict
- **Real embeddings transform retrieval.** `vector` goes from worst to best, and RRF fusion now
  earns its place (L1 fused 0.75 is the top single-file result).
- On Level-2, **pure `vector` is best (Hit@k 0.95)** — keyword slightly drags the fusion down there.
- **Best configuration now:** `--embeddings gemini --no-query-rewrite`, modes `vector,fused`
  (keep keyword for the comparison table). This is what the full generation experiment should use.

Run it with, e.g.:
```
python src/RAG.py --docs dataset_json --qa-csv tablev2/QA_test.csv \
  --output-csv result/QA_test_out.csv --no-query-rewrite --embeddings gemini --modes keyword,vector,fused
```

## 7. Full generation run — best config (Gemini embeddings, no rewrite) ✅

Ran all 20+20 questions with generation: `--embeddings gemini --no-query-rewrite --modes
keyword,vector,fused`. Outputs: `result/QA_test_out_gemini.csv`, `result/QA_test_lvl2_out_gemini.csv`.

**Retrieval Hit@k** — keyword 0.45 / 0.75, vector 0.65 / **0.95**, fused **0.75** / 0.85 (L1 / L2).

**Rubric scores (fused mode), vs. the earlier hash-embedding baseline:**

| Dimension | L1 (old → new) | L2 (old → new) |
|---|---|---|
| Engineering Correctness /40 | 19.6 → **33.0** | 22.4 → **32.9** |
| Document Grounding /30 | 19.4 → **23.1** | 20.6 → **23.4** |
| Response Latency /10 | 0.0 → 3.2 | 0.0 → 0.8 |
| Bilingual /10 | 9.0 | 9.0 |
| Prompt Compliance /10 | 9.0 | 9.0 |
| **Total /100** | 57.0 → **77.3** | 61.0 → **75.0** |

Real embeddings + dropping query-rewrite lifted both files ~16–20 points. Latency is now the
weakest dimension (L1 mean 19.7 s, L2 28.7 s). The paper's §IV (Table II retrieval, Table III
rubric) was updated to these numbers.

**Bug fixed during this run:** the `_reformat_answer` success `print` used `→`/`—`, which crashed
on the Windows cp1252 console and made the code discard every reformatted answer. Replaced with
ASCII and set `sys.stdout`/`stderr` to UTF-8 in `main()`. (First-pass answers were already clean,
so the discarded reformat did not materially change quality.)

---

*Outputs: `result/result_pilot_lvl1.csv`, `result/result_pilot_lvl1_retrieval.csv`, the four
`result/sweep_*_rewrite/norewrite.csv` (rewrite A/B), `result/sweep_lvl{1,2}_gemini.csv`
(embedding comparison), and `result/QA_test{,_lvl2}_out_gemini.csv` (full scored run). Chunk
embeddings cached in `.emb_cache/`.*
