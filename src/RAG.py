"""RAG pipeline implementing the POLRAG framework.

Three stages:
1) Knowledge Base Construction: OCR JSON → chapter-aware chunking → vector embedding
2) Retrieval: Query rewriting → BM25 keyword + semantic similarity → RRF fusion
3) Generation: Structured prompt (system role + context + guidance) + multi-turn history

Usage:
    python src/RAG.py --docs dataset_json --query "เกณฑ์การซ่อมแซมรอยร้าว (Crack Repair) สำหรับรอยร้าวแบบ Hard Crack ที่ต้องซ่อมแซมด้วย Cementitious Mortar คืออะไร"
    python src/RAG.py --docs dataset_json --interactive
    python src/RAG.py --docs dataset_json --qa-csv tablev2/QA_test_lvl2.csv --output-csv result_test_lvl2.csv
    python src/RAG.py --docs dataset_json --qa-csv tablev2/QA_test.csv --output-csv result_test.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

try:
    from google import genai
except ImportError:
    genai = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


DEFAULT_QUERY = ""

SYSTEM_ROLE = (
    "You are an expert technical assistant specializing in tunnel segment repair methods. "
    "Answer questions strictly based on the provided document context. "
    "Always cite the chapter and page number when referencing specific information. "
    "If the context does not contain enough information, state that clearly rather than guessing."
)


def load_api_keys(gemini_api_key: str | None = None, verbose: bool = False) -> list[str]:
    """Resolve Gemini API keys: geminikey.txt (next to RAG.py, then project root) → env var → CLI arg."""
    here = Path(__file__).parent
    file_keys: list[str] = []
    for keys_file in (here / "geminikey.txt", here.parent / "geminikey.txt"):
        if keys_file.exists():
            file_keys = [k.strip() for k in keys_file.read_text(encoding="utf-8").splitlines() if k.strip()]
            if verbose:
                print(f"  Loaded {len(file_keys)} API keys from {keys_file}")
            break
    env_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return file_keys or ([env_key] if env_key else [])


def ping_gemini(keys: list[str], model: str = "gemini-2.5-flash") -> tuple[bool, str]:
    """Send one minimal request to verify the API is reachable. No KB build required."""
    if genai is None:
        return False, "package 'google-genai' is not installed"
    if not keys:
        return False, "no Gemini API key found (geminikey.txt or GOOGLE_API_KEY)"
    try:
        client = genai.Client(api_key=keys[0])
        resp = client.models.generate_content(model=model, contents="ping")
        text = getattr(resp, "text", None)
        return True, f"reachable ({model}); sample reply: {(text or '').strip()[:40]!r}"
    except Exception as exc:  # noqa: BLE001 — surface any auth/quota/network error verbatim
        return False, f"{type(exc).__name__}: {exc}"


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings used when no OpenAI key is available."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+|[฀-๿]+", text.lower())

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


class GeminiEmbeddings(Embeddings):
    """Real semantic embeddings via the Gemini embedding API, with an on-disk cache.

    Reuses the project's Gemini API keys (rotating on rate limits) and caches each
    text's vector by SHA-256 so repeated runs do not re-embed the same chunks.
    """

    BATCH = 100

    def __init__(self, api_keys: list[str], model: str = "gemini-embedding-001",
                 cache_path: str | Path | None = None) -> None:
        if genai is None:
            raise ImportError("Missing package 'google-genai'. Install: pip install google-genai")
        if not api_keys:
            raise ValueError("GeminiEmbeddings requires at least one API key.")
        self.api_keys = api_keys
        self.model = model
        self._ki = 0
        self.client = genai.Client(api_key=api_keys[0])
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                print(f"  Embedding cache: {len(self.cache)} vectors from {self.cache_path.name}")
            except Exception:
                self.cache = {}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()

    def _rotate(self) -> None:
        self._ki = (self._ki + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self._ki])

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last = ""
        for _ in range(len(self.api_keys) + 1):
            try:
                resp = self.client.models.embed_content(model=self.model, contents=texts)
                return [list(e.values) for e in resp.embeddings]
            except Exception as exc:
                last = str(exc)
                if any(s in last for s in ("RESOURCE_EXHAUSTED", "RATE_LIMIT", "429", "quota")):
                    self._rotate()
                    time.sleep(1.0)
                else:
                    raise
        raise RuntimeError(f"Gemini embedding failed: {last}")

    def _save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        todo: list[str] = []
        todo_idx: list[int] = []
        for i, t in enumerate(texts):
            cached = self.cache.get(self._key(t))
            if cached is not None:
                out[i] = cached
            else:
                todo.append(t)
                todo_idx.append(i)
        for b in range(0, len(todo), self.BATCH):
            batch = todo[b:b + self.BATCH]
            for j, vec in zip(todo_idx[b:b + self.BATCH], self._embed_batch(batch)):
                out[j] = vec
                self.cache[self._key(texts[j])] = vec
        if todo:
            self._save()
        return [v for v in out]  # type: ignore[misc]

    def embed_query(self, text: str) -> list[float]:
        cached = self.cache.get(self._key(text))
        if cached is not None:
            return cached
        vec = self._embed_batch([text])[0]
        self.cache[self._key(text)] = vec
        self._save()
        return vec


class RAGFlow:
    # BM25 hyperparameters (standard values per Robertson et al. 2009)
    BM25_K1 = 1.5
    BM25_B = 0.75

    DEFAULT_MODES = ("keyword", "vector", "fused")

    def __init__(
        self,
        documents_dir: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        gemini_model: str = "gemini-2.5-flash",
        gemini_api_key: str | None = None,
        top_k_each: int = 8,
        top_k_fused: int = 5,
        rrf_k: int = 60,
        enable_query_rewrite: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        enable_reformat: bool = True,
        embedding_dim: int = 512,
        modes: list[str] | None = None,
        require_gemini: bool = True,
        embeddings_backend: str = "auto",
        gemini_embed_model: str = "gemini-embedding-001",
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.gemini_model = gemini_model
        self.top_k_each = top_k_each
        self.top_k_fused = top_k_fused
        self.rrf_k = rrf_k
        self.enable_query_rewrite = enable_query_rewrite
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.enable_reformat = enable_reformat
        self.embedding_dim = embedding_dim
        self.modes = list(modes) if modes else list(self.DEFAULT_MODES)
        self.conversation_history: list[dict[str, str]] = []

        # Generation needs google-genai + a key; retrieval-only mode does not.
        if genai is None and require_gemini:
            raise ImportError("Missing package 'google-genai'. Install: pip install google-genai")

        # Load API keys: geminikey.txt (project root) → env var → CLI arg
        all_keys = load_api_keys(gemini_api_key, verbose=True)
        if not all_keys and require_gemini:
            raise ValueError("No Gemini API key found. Add keys to geminikey.txt or set GOOGLE_API_KEY.")

        self._api_keys: list[str] = all_keys
        self._key_index: int = 0
        self.client = genai.Client(api_key=all_keys[0]) if (genai is not None and all_keys) else None

        openai_key = os.getenv("OPENAI_API_KEY")
        backend = embeddings_backend
        if backend == "auto":
            backend = "openai" if openai_key else "hash"
        if backend == "openai":
            if not openai_key:
                raise ValueError("--embeddings openai requires OPENAI_API_KEY.")
            self.embeddings: Embeddings = OpenAIEmbeddings(api_key=openai_key)
        elif backend == "gemini":
            cache = Path(__file__).parent.parent / ".emb_cache" / f"{gemini_embed_model}.json"
            self.embeddings = GeminiEmbeddings(self._api_keys, model=gemini_embed_model, cache_path=cache)
        else:  # "hash" — deterministic placeholder, no semantics
            self.embeddings = LocalHashEmbeddings(dim=self.embedding_dim)
        print(f"  Embeddings backend: {backend}")

        print("Building knowledge base...")
        self.documents = self._load_and_chunk_documents()
        print(f"  Loaded {len(self.documents)} chunks from {self.documents_dir}")
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.doc_tokens = [self._tokenize(doc.page_content) for doc in self.documents]
        self._bm25 = self._build_bm25_stats(self.doc_tokens)
        print("Knowledge base ready.\n")

    # ── Knowledge Base Construction (POLRAG §III.B) ──────────────────────────

    def _iter_source_files(self) -> list[Path]:
        if not self.documents_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.documents_dir}")
        files = [
            p for p in self.documents_dir.rglob("*.json")
            if p.is_file()
        ]
        if not files:
            raise ValueError(f"No JSON files found in {self.documents_dir}")
        return sorted(files)

    def _extract_page_text(self, file_path: Path) -> str:
        """Extract full-page OCR text from the JSON format.

        The JSON is a list of OCR tokens. The first element always contains
        the entire page text in its 'description' field.
        """
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and "description" in first:
                return str(first["description"])
        return raw

    def _chapter_from_path(self, file_path: Path) -> str:
        """Convert folder name to a readable chapter title.

        e.g. '3_Repair_With_Epoxy_Resin' → 'Repair With Epoxy Resin'
        """
        folder = file_path.parent.name
        parts = folder.split("_")
        if parts and parts[0].isdigit():
            parts = parts[1:]
        return " ".join(parts)

    def _load_and_chunk_documents(self) -> list[Document]:
        base_docs: list[Document] = []
        for file_path in self._iter_source_files():
            text = self._extract_page_text(file_path)
            if not text.strip():
                continue
            page_num = self._page_num_from_path(str(file_path))
            chapter = self._chapter_from_path(file_path)
            base_docs.append(Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "chapter": chapter,
                    "page": page_num,
                },
            ))

        # POLRAG targets 200–300 words per chunk; ~1500 chars ≈ 300 words
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(base_docs)

    # ── BM25 Keyword Retrieval (POLRAG §III.C) ───────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+|[฀-๿]+", text.lower())

    def _build_bm25_stats(self, tokenized_docs: list[list[str]]) -> dict:
        n = len(tokenized_docs)
        df: dict[str, int] = {}
        doc_lengths: list[int] = []
        for tokens in tokenized_docs:
            doc_lengths.append(len(tokens))
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        avgdl = sum(doc_lengths) / max(1, n)
        idf = {
            token: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in df.items()
        }
        return {"idf": idf, "doc_lengths": doc_lengths, "avgdl": avgdl}

    def _bm25_retrieval(self, query: str, top_k: int) -> list[int]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        idf = self._bm25["idf"]
        avgdl = self._bm25["avgdl"]
        doc_lengths = self._bm25["doc_lengths"]
        k1, b = self.bm25_k1, self.bm25_b

        scores: list[tuple[int, float]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            dl = doc_lengths[idx]
            score = 0.0
            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                score += idf.get(qt, 0.0) * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / avgdl)
                )
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]

    # ── Semantic Similarity Retrieval (POLRAG §III.C) ────────────────────────

    def _vector_retrieval(self, query: str, top_k: int) -> list[int]:
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        index_by_key = {self._doc_key(doc): i for i, doc in enumerate(self.documents)}
        ranked: list[tuple[int, float]] = []
        for doc, distance in results:
            idx = index_by_key.get(self._doc_key(doc))
            if idx is not None:
                ranked.append((idx, float(distance)))
        ranked.sort(key=lambda x: x[1])
        return [idx for idx, _ in ranked]

    @staticmethod
    def _doc_key(doc: Document) -> tuple[str, str]:
        return (str(doc.metadata.get("source", "")), doc.page_content)

    # ── RRF Fusion (POLRAG §III.C) ───────────────────────────────────────────

    def _rrf_fusion(self, rankings: list[list[int]], top_k: int) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_idx in enumerate(ranking, start=1):
                fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (self.rrf_k + rank)
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ordered[:top_k]]

    # ── Query Optimization (POLRAG §III.D) ──────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        """Rewrite user question into a structured retrieval expression via LLM."""
        prompt = (
            "Convert the following user question into a structured search expression "
            "for technical document retrieval about tunnel segment repair.\n"
            "Output format: subject: <subject>; elements: <element1>, <element2>, ...\n\n"
            "Examples:\n"
            "Q: How do I fix cracks using epoxy?\n"
            "A: subject: crack repair; elements: epoxy resin, procedure, application steps\n\n"
            "Q: วิธีซ่อมโพรงอากาศคืออะไร\n"
            "A: subject: void and blowhole repair; elements: repair method, materials, steps\n\n"
            f"Q: {query}\n"
            "A:"
        )
        try:
            text, _ = self._generate_with_gemini(prompt)
            rewritten = text.strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    # ── Prompt Engineering (POLRAG §III.E) ──────────────────────────────────

    def _build_context(self, doc_ids: list[int]) -> str:
        """Format retrieved chunks with chapter and page citation."""
        parts: list[str] = []
        for i, doc_idx in enumerate(doc_ids, start=1):
            doc = self.documents[doc_idx]
            chapter = doc.metadata.get("chapter", "Unknown Chapter")
            page = doc.metadata.get("page", "?")
            parts.append(f"[{i}] Chapter: {chapter}, Page: {page}\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    def _build_polrag_prompt(self, query: str, context: str) -> str:
        """Three-stage POLRAG prompt: system role + context + answer guidance."""
        history_text = ""
        if self.conversation_history:
            recent = self.conversation_history[-4:]
            turns = [f"User: {t['user']}\nAssistant: {t['assistant']}" for t in recent]
            history_text = "\n\n".join(turns) + "\n\n"

        return (
            f"{SYSTEM_ROLE}\n\n"
            f"=== Document Context ===\n{context}\n\n"
            + (f"=== Conversation History ===\n{history_text}" if history_text else "")
            + f"=== Question ===\n{query}\n\n"
            "=== Answer Guidelines ===\n"
            "1. Use numbered steps for procedures; bullet points for lists.\n"
            "2. Answer in the same language as the question.\n"
            "3. If context is insufficient, say so explicitly.\n\n"
            "=== Required Output Format (do not deviate) ===\n"
            "Page: <comma-separated page numbers referenced, e.g. 12, 15>\n"
            "Answer: <your full answer here>\n"
        )

    @staticmethod
    def _parse_response(raw: str, fallback_page: str) -> tuple[str, str]:
        """Extract Page and Answer fields from the structured LLM response."""
        page_match = re.search(r"(?:^|\n)\s*Page\s*:\s*(.+)", raw, flags=re.IGNORECASE)
        answer_match = re.search(r"(?:^|\n)\s*Answer\s*:\s*([\s\S]+)", raw, flags=re.IGNORECASE)

        page = page_match.group(1).strip() if page_match else fallback_page
        answer = answer_match.group(1).strip() if answer_match else raw.strip()
        return page, answer

    # ── Answer Reformatting ──────────────────────────────────────────────────

    def _reformat_answer(self, question: str, answer: str) -> str:
        """Second LLM pass to extract relevant content and reformat the answer cleanly.

        The input may be noisy OCR text containing document headers, company names,
        project titles, and table fragments. This pass filters all of that out and
        returns only a clean, human-readable answer to the question.
        """
        prompt = (
            "You are a technical writing assistant for construction documents.\n"
            "The text below may be raw OCR output from scanned PDF pages. "
            "It likely contains noise such as:\n"
            "  - Company names (e.g. STECON, SINO-THAI ENGINEERING)\n"
            "  - Project titles (e.g. SEWERAGE TUNNEL BUNG NONG BON)\n"
            "  - Document headers, page numbers, chapter labels\n"
            "  - Broken table fragments and OCR artifacts\n"
            "  - Repeated or duplicate lines\n\n"
            "Your job:\n"
            "1. IGNORE all company names, project titles, document headers, and OCR noise.\n"
            "2. EXTRACT only the technical content that directly answers the question.\n"
            "3. Rewrite it as a clean, concise answer in the same language as the question.\n"
            "4. Use numbered steps for procedures; bullet points for non-sequential lists.\n"
            "5. Keep ALL technical values, numbers, units, and material names intact.\n"
            "6. If no relevant technical content is found, reply: "
            "'ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร / No relevant information found in the document.'\n"
            "7. Do NOT invent or add any information not present in the text.\n\n"
            f"Question: {question}\n\n"
            f"Raw Text:\n{answer}\n\n"
            "Clean Answer:"
        )
        try:
            time.sleep(1.0)  # brief pause to avoid rate-limiting after the main generation call
            reformatted, _ = self._generate_with_gemini(prompt)
            result = reformatted.strip() if reformatted else ""
            if not result:
                print("[reformat] WARNING: LLM returned empty response, keeping original.")
                return answer
            print(f"[reformat] OK - reformatted {len(answer)} chars -> {len(result)} chars")
            return result
        except Exception as exc:
            print(f"[reformat] ERROR: {exc}. Keeping original answer.")
            return answer

    # ── Generation ───────────────────────────────────────────────────────────

    def _rotate_api_key(self) -> None:
        """Advance to the next API key in the pool and recreate the client."""
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        self.client = genai.Client(api_key=self._api_keys[self._key_index])
        print(f"[key-rotate] Switched to key index {self._key_index} / {len(self._api_keys) - 1}")

    _RATE_LIMIT_SIGNALS = ("RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED", "429", "quota")

    def _generate_with_gemini(self, prompt: str) -> tuple[str, str]:
        candidate_models = [self.gemini_model, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
        seen: set[str] = set()
        model_list: list[str] = []
        for m in candidate_models:
            if m and m not in seen:
                seen.add(m)
                model_list.append(m)

        last_error = ""
        for model_name in model_list:
            # Try every available API key before moving to the next model
            for _ in range(len(self._api_keys)):
                try:
                    response = self.client.models.generate_content(model=model_name, contents=prompt)
                    text = getattr(response, "text", None)
                    return (text.strip() if text else str(response)), model_name
                except Exception as exc:
                    last_error = str(exc)
                    if any(sig in last_error for sig in self._RATE_LIMIT_SIGNALS):
                        print(f"[key-rotate] Rate limit on key {self._key_index} ({model_name}), rotating…")
                        self._rotate_api_key()
                        time.sleep(1.0)
                    else:
                        break  # Non-rate-limit error — skip to next model immediately

        raise RuntimeError(last_error or "Gemini generation failed")

    def _extractive_fallback(self, query: str, context: str, max_sentences: int = 5) -> str:
        q_tokens = set(self._tokenize(query))
        chunks = re.split(r"(?<=[.!?\n])\s+", context)
        scored: list[tuple[float, str]] = []
        for sent in chunks:
            clean = sent.strip()
            if len(clean) < 20:
                continue
            s_tokens = self._tokenize(clean)
            overlap = sum(1 for t in s_tokens if t in q_tokens)
            if overlap > 0:
                scored.append((overlap / max(1, len(set(s_tokens))), clean))
        if not scored:
            return context[:900].strip()
        scored.sort(reverse=True)
        return "\n".join(s for _, s in scored[:max_sentences])

    # ── Public API ───────────────────────────────────────────────────────────

    def ask(self, query: str) -> dict[str, Any]:
        """Single-turn or multi-turn question answering following the POLRAG flow."""
        # Stage 2a: Query rewriting
        retrieval_query = self._rewrite_query(query) if self.enable_query_rewrite else query

        # Stage 2b: Multi-channel recall
        bm25_rank = self._bm25_retrieval(retrieval_query, top_k=self.top_k_each)
        vector_rank = self._vector_retrieval(retrieval_query, top_k=self.top_k_each)

        # Stage 2c: RRF fusion
        fused_ids = self._rrf_fusion([bm25_rank, vector_rank], top_k=self.top_k_fused)

        # Stage 3: Generation with structured prompt
        context = self._build_context(fused_ids)
        prompt = self._build_polrag_prompt(query, context)

        # Page comes from retrieved chunk metadata, not from LLM output text
        page = self._pages_from_doc_ids(fused_ids)

        llm_model_used = ""
        try:
            raw_answer, llm_model_used = self._generate_with_gemini(prompt)
        except Exception as exc:
            err_text = str(exc)
            if any(c in err_text for c in ("RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED", "429")):
                raw_answer = self._extractive_fallback(query, context)
                llm_model_used = "extractive-fallback"
            else:
                raise

        _, answer = self._parse_response(raw_answer, page)
        answer = self._reformat_answer(query, answer)
        self.conversation_history.append({"user": query, "assistant": answer})

        return {
            "query": query,
            "rewritten_query": retrieval_query,
            "retrieval": {
                "bm25_rank": bm25_rank,
                "vector_rank": vector_rank,
                "fused_rank": fused_ids,
            },
            "context": context,
            "answer_model": llm_model_used,
            "page": page,
            "answer": answer,
        }

    def reset_conversation(self) -> None:
        self.conversation_history.clear()

    # ── CSV Batch Mode ───────────────────────────────────────────────────────

    def _retrieve_ranks(self, retrieval_query: str) -> dict[str, list[int]]:
        """Run the base retrievers once and return the doc-id list for each requested mode."""
        bm25_rank = self._bm25_retrieval(retrieval_query, top_k=self.top_k_each)
        vector_rank = self._vector_retrieval(retrieval_query, top_k=self.top_k_each)
        fused_ids = self._rrf_fusion([bm25_rank, vector_rank], top_k=self.top_k_fused)

        # Each mode uses its own retrieved document set for context and generation.
        # NOTE: "keyword" and "fulltext" are both BM25 (term-based) and therefore identical today.
        all_modes = {
            "keyword":  bm25_rank[:self.top_k_fused],
            "fulltext": bm25_rank[:self.top_k_fused],
            "vector":   vector_rank[:self.top_k_fused],
            "fused":    fused_ids,
        }
        ranks = {m: all_modes[m] for m in self.modes if m in all_modes}
        ranks["_bm25_rank"] = bm25_rank
        ranks["_vector_rank"] = vector_rank
        ranks["_fused_rank"] = fused_ids
        return ranks

    def predict_expected_columns(self, question: str) -> dict[str, Any]:
        """Batch prediction mode compatible with the existing CSV evaluation format."""
        retrieval_query = self._rewrite_query(question) if self.enable_query_rewrite else question
        ranks = self._retrieve_ranks(retrieval_query)
        bm25_rank, vector_rank, fused_ids = ranks["_bm25_rank"], ranks["_vector_rank"], ranks["_fused_rank"]
        mode_doc_ids = {m: ranks[m] for m in self.modes if m in ranks}

        by_mode: dict[str, Any] = {}
        for mode, doc_ids in mode_doc_ids.items():
            context = self._build_context(doc_ids)
            prompt = self._build_polrag_prompt(question, context)

            # Page comes from retrieved chunk metadata, not from LLM output text
            page = self._pages_from_doc_ids(doc_ids)

            try:
                raw_answer, model_used = self._generate_with_gemini(prompt)
            except Exception as exc:
                # Any generation failure (rate limit, dead model, network) falls back to
                # extractive answering so a single bad question cannot abort the batch.
                print(f"[generate] {mode} fallback after error: {str(exc)[:80]}")
                raw_answer = self._extractive_fallback(question, context)
                model_used = "extractive-fallback"

            parsed = self._extract_json_object(raw_answer) or self._extract_labeled_fields(raw_answer)
            predictions = self._normalize_prediction_payload(parsed, raw_answer, question, page)
            predictions["predicted_Page"] = page   # override with metadata-derived pages
            if self.enable_reformat:
                predictions["predicted_Answer"] = self._reformat_answer(question, predictions["predicted_Answer"])
            by_mode[mode] = {"answer_model": model_used, "raw_answer": raw_answer, "predictions": predictions}

        return {
            "question": question,
            "rewritten_query": retrieval_query,
            "retrieval": {"bm25_rank": bm25_rank, "vector_rank": vector_rank, "fused_rank": fused_ids},
            "by_mode": by_mode,
        }

    # ── Static Helpers ───────────────────────────────────────────────────────

    def _pages_from_doc_ids(self, doc_ids: list[int]) -> str:
        """Derive page references directly from retrieved document metadata.

        More reliable than parsing page numbers from LLM output text, because
        the metadata is set at index time from the source file path.
        Returns a comma-separated string of unique page numbers in retrieval order.
        """
        seen: set[str] = set()
        pages: list[str] = []
        for i in doc_ids:
            page = str(self.documents[i].metadata.get("page", "")).strip()
            if page and page not in seen:
                seen.add(page)
                pages.append(page)
        return ", ".join(pages) if pages else "unknown"

    @staticmethod
    def _page_num_from_path(source: str) -> str:
        match = re.search(r"page[_-]?(\d+)", source, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    # ── Per-method Evaluation Scoring ────────────────────────────────────────

    @staticmethod
    def _parse_pages(value: Any) -> set[str]:
        """Normalise a page field ('6, 8, 0135' or '3.0' or 'unknown') into a set of ints-as-str."""
        pages: set[str] = set()
        for tok in re.split(r"[,\s]+", str(value).strip()):
            tok = tok.strip()
            if not tok or tok.lower() in ("unknown", "nan", "none"):
                continue
            m = re.match(r"0*(\d+)", tok)  # strip leading zeros / trailing '.0'
            if m:
                pages.add(str(int(m.group(1))))
        return pages

    @classmethod
    def _page_scores(cls, pred_page: Any, gt_page: Any) -> tuple[float, float, int]:
        """Return (recall, precision, hit) of retrieved pages against the GT page set."""
        pred, gt = cls._parse_pages(pred_page), cls._parse_pages(gt_page)
        if not gt:
            return (0.0, 0.0, 0)
        inter = pred & gt
        recall = len(inter) / len(gt)
        precision = len(inter) / len(pred) if pred else 0.0
        return (recall, precision, 1 if inter else 0)

    def _answer_similarity(self, pred: str, gt: str) -> float:
        """Cosine similarity between predicted and gold answer embeddings (relative measure)."""
        pred, gt = (pred or "").strip(), (gt or "").strip()
        if not pred or not gt:
            return 0.0
        try:
            a = self.embeddings.embed_query(pred)
            b = self.embeddings.embed_query(gt)
        except Exception:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        match = re.search(r"\{.*?\}", fenced, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_labeled_fields(text: str) -> dict[str, str]:
        cleaned = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

        def capture(pattern: str) -> str:
            m = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        return {
            "Rainbow Group": capture(r"Rainbow Group\s*:\s*(.+)"),
            "Category": capture(r"Category\s*:\s*(.+)"),
            "Answer": capture(r"Answer\s*:\s*(.*?)(?:\n\s*Page\s*:|\Z)"),
            "Page": capture(r"Page\s*:\s*(.+)"),
        }

    @staticmethod
    def _infer_category(question: str, answer: str) -> str:
        text = f"{question} {answer}".lower()
        if any(t in text for t in ["risk", "safety", "อันตราย", "chemical", "สารเคมี"]):
            return "Human Risk"
        if any(t in text for t in ["environment", "preventive", "สิ่งแวดล้อม"]):
            return "Environmental / Preventive Measures"
        if any(t in text for t in ["material", "mixing ratio", "compressive", "วัสดุ"]):
            return "Material"
        if any(t in text for t in ["repair", "crack", "spalling", "blowholes", "ซ่อม", "รอยร้าว"]):
            return "Repair Method"
        return "General / Admin"

    @staticmethod
    def _infer_rainbow_group(category: str) -> str:
        if category in {"Repair Method", "Material", "Human Risk", "Environmental / Preventive Measures"}:
            return "Construction-Related"
        return "Non-Construction Questions"

    @staticmethod
    def _normalize_prediction_payload(
        payload: dict[str, Any], raw_answer: str, question: str, fallback_page: str
    ) -> dict[str, str]:
        def get(*keys: str) -> str:
            for k in keys:
                if k in payload and payload[k]:
                    return str(payload[k]).strip()
            return ""

        category = get("Category", "category") or RAGFlow._infer_category(question, raw_answer)
        rainbow_group = get("Rainbow Group", "rainbow_group") or RAGFlow._infer_rainbow_group(category)
        answer = get("Answer", "answer") or raw_answer.strip() or "unknown"
        page = get("Page", "page") or fallback_page or "unknown"

        return {
            "predicted_Rainbow Group": rainbow_group,
            "predicted_Category": category,
            "predicted_Answer": answer,
            "predicted_Page": page,
        }


def summarize_modes(frame: "pd.DataFrame", modes: list[str]) -> dict[str, dict[str, float]]:
    """Mean page-hit / answer-similarity per mode, read back from a scored results frame."""
    def col_mean(col: str) -> float:
        if col not in frame.columns:
            return float("nan")
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else float("nan")

    summary: dict[str, dict[str, float]] = {}
    for mode in modes:
        summary[mode] = {
            "recall": col_mean(f"predicted_{mode}_PageRecall"),
            "prec":   col_mean(f"predicted_{mode}_PagePrec"),
            "hit":    col_mean(f"predicted_{mode}_Hit"),
            "sim":    col_mean(f"predicted_{mode}_AnsSim"),
        }
    return summary


def format_mode_table(summary: dict[str, dict[str, float]], include_sim: bool = True) -> str:
    """Render the per-method comparison as a markdown table (single methods vs fused)."""
    cols = ["Mode", "PageRecall", "PagePrec", "Hit@k"] + (["AnsSim"] if include_sim else [])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for mode, s in summary.items():
        tag = mode + (" (combined)" if mode == "fused" else " (single)")
        row = [tag, f"{s['recall']:.3f}", f"{s['prec']:.3f}", f"{s['hit']:.3f}"]
        if include_sim:
            row.append(f"{s['sim']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POLRAG: RAG-LLM framework for document QA.")
    parser.add_argument("--docs", required=True, help="Directory containing source JSON docs.")
    parser.add_argument("--query", default=None, help="Single user question.")
    parser.add_argument("--interactive", action="store_true", help="Multi-turn conversation mode.")
    parser.add_argument("--qa-csv", default=None, help="CSV file with questions for batch evaluation.")
    parser.add_argument("--output-csv", default=None, help="Output CSV path for batch predictions.")
    parser.add_argument("--question-col", default="Question", help="Question column name in --qa-csv.")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size in characters (~300 words).")
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-k-each", type=int, default=8)
    parser.add_argument("--top-k-fused", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 term-frequency saturation (k1).")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 length-normalisation (b).")
    parser.add_argument("--embedding-dim", type=int, default=512,
                        help="Local hash-embedding dimension (only used by --embeddings hash).")
    parser.add_argument("--embeddings", choices=["auto", "gemini", "openai", "hash"], default="auto",
                        help="Embedding backend: gemini (real semantics, reuses keys), openai "
                             "(needs OPENAI_API_KEY), hash (placeholder), auto = openai-if-keyed-else-hash.")
    parser.add_argument("--modes", default="keyword,vector,fused",
                        help="Comma list of retrieval modes to run: keyword,fulltext,vector,fused. "
                             "(keyword and fulltext are identical BM25 today.)")
    parser.add_argument("--no-query-rewrite", action="store_true", help="Skip LLM query rewriting.")
    parser.add_argument("--no-reformat", action="store_true", help="Skip the second LLM answer-reformatting pass.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions of --qa-csv.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Score retrieval (page-hit) per mode without any Gemini generation.")
    parser.add_argument("--check-gemini", action="store_true",
                        help="Ping the Gemini API and exit (no knowledge base built).")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context before answer.")
    return parser.parse_args()


def main() -> None:
    # Avoid Windows console (cp1252/charmap) crashes on Thai text and symbols like → —.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    # ── Gemini availability check (no knowledge base built) ───────────────────
    if args.check_gemini:
        ok, msg = ping_gemini(load_api_keys(args.gemini_api_key, verbose=True), args.gemini_model)
        print(f"Gemini API: {'OK — ' if ok else 'UNAVAILABLE — '}{msg}")
        raise SystemExit(0 if ok else 1)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rag = RAGFlow(
        documents_dir=args.docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        gemini_model=args.gemini_model,
        gemini_api_key=args.gemini_api_key,
        top_k_each=args.top_k_each,
        top_k_fused=args.top_k_fused,
        rrf_k=args.rrf_k,
        enable_query_rewrite=not args.no_query_rewrite,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        enable_reformat=not args.no_reformat,
        embedding_dim=args.embedding_dim,
        modes=modes,
        embeddings_backend=args.embeddings,
        # Gemini is needed for generation, for query rewriting, or for Gemini embeddings.
        require_gemini=(not args.retrieval_only) or (not args.no_query_rewrite) or (args.embeddings == "gemini"),
    )

    # ── Batch CSV mode ───────────────────────────────────────────────────────
    if args.qa_csv:
        qa_csv_path = Path(args.qa_csv)
        if not qa_csv_path.exists():
            raise FileNotFoundError(f"QA CSV not found: {qa_csv_path}")

        frame = pd.read_csv(qa_csv_path)
        if args.question_col not in frame.columns:
            raise ValueError(
                f"Column '{args.question_col}' not found. Available: {list(frame.columns)}"
            )
        if args.limit is not None:
            frame = frame.head(args.limit).copy()
            print(f"  --limit {args.limit}: running first {len(frame)} questions only.")

        has_gt_page = "Page" in frame.columns
        has_gt_answer = "Answer" in frame.columns

        # ── Retrieval-only: page-hit metrics per mode, no Gemini generation ───
        if args.retrieval_only:
            score_cols = {f"predicted_{m}_{c}": [] for m in modes for c in ("Page", "PageRecall", "PagePrec", "Hit")}
            for idx, row in frame.iterrows():
                question = str(row[args.question_col]).strip()
                gt_page = row["Page"] if has_gt_page else ""
                if question and question.lower() != "nan":
                    rq = rag._rewrite_query(question) if rag.enable_query_rewrite else question
                    ranks = rag._retrieve_ranks(rq)
                else:
                    ranks = {}
                for mode in modes:
                    pages = rag._pages_from_doc_ids(ranks[mode]) if mode in ranks else ""
                    recall, prec, hit = RAGFlow._page_scores(pages, gt_page)
                    score_cols[f"predicted_{mode}_Page"].append(pages)
                    score_cols[f"predicted_{mode}_PageRecall"].append(round(recall, 4))
                    score_cols[f"predicted_{mode}_PagePrec"].append(round(prec, 4))
                    score_cols[f"predicted_{mode}_Hit"].append(hit)
                print(f"Retrieved {idx + 1}/{len(frame)}")
            for col, vals in score_cols.items():
                frame[col] = vals
            summary = summarize_modes(frame, modes)
            print("\n=== Retrieval-only comparison (page-hit) ===")
            print(format_mode_table(summary, include_sim=False))
            if args.output_csv:
                frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
                print(f"\nSaved retrieval scores to: {args.output_csv}")
            return

        cols = ["Rainbow Group", "Category", "Answer", "Page", "answer_model", "raw_answer",
                "PageRecall", "PagePrec", "Hit", "AnsSim"]
        predicted_columns: dict[str, list[Any]] = {
            f"predicted_{mode}_{col}": [] for mode in modes for col in cols
        }

        batch_start = time.time()
        row_times: list[float] = []

        for idx, row in frame.iterrows():
            question = str(row[args.question_col]).strip()
            gt_answer = str(row["Answer"]) if has_gt_answer else ""
            gt_page = row["Page"] if has_gt_page else ""
            if not question or question.lower() == "nan":
                for col in predicted_columns:
                    predicted_columns[col].append("")
                row_times.append(0.0)
                continue

            q_start = time.time()
            result = rag.predict_expected_columns(question)
            q_elapsed = time.time() - q_start
            row_times.append(q_elapsed)

            for mode in modes:
                preds = result["by_mode"][mode]["predictions"]
                recall, prec, hit = RAGFlow._page_scores(preds["predicted_Page"], gt_page)
                sim = rag._answer_similarity(preds["predicted_Answer"], gt_answer)
                predicted_columns[f"predicted_{mode}_Rainbow Group"].append(preds["predicted_Rainbow Group"])
                predicted_columns[f"predicted_{mode}_Category"].append(preds["predicted_Category"])
                predicted_columns[f"predicted_{mode}_Answer"].append(preds["predicted_Answer"])
                predicted_columns[f"predicted_{mode}_Page"].append(preds["predicted_Page"])
                predicted_columns[f"predicted_{mode}_answer_model"].append(result["by_mode"][mode]["answer_model"])
                predicted_columns[f"predicted_{mode}_raw_answer"].append(result["by_mode"][mode]["raw_answer"])
                predicted_columns[f"predicted_{mode}_PageRecall"].append(round(recall, 4))
                predicted_columns[f"predicted_{mode}_PagePrec"].append(round(prec, 4))
                predicted_columns[f"predicted_{mode}_Hit"].append(hit)
                predicted_columns[f"predicted_{mode}_AnsSim"].append(round(sim, 4))
            print(f"Processed {idx + 1}/{len(frame)} — {q_elapsed:.1f}s (total so far: {time.time() - batch_start:.1f}s)")

        batch_elapsed = time.time() - batch_start
        avg_time = batch_elapsed / max(len(row_times), 1)
        print(
            f"\nBatch complete — {len(frame)} questions in {batch_elapsed:.1f}s "
            f"(avg {avg_time:.1f}s / question)"
        )

        for col, values in predicted_columns.items():
            frame[col] = values
        frame["compute_time_s"] = [round(t, 2) for t in row_times]

        out_path = Path(args.output_csv) if args.output_csv else qa_csv_path.with_name(
            f"{qa_csv_path.stem}_predicted{qa_csv_path.suffix}"
        )
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved predictions to: {out_path}")

        summary = summarize_modes(frame, modes)
        print(f"\n=== Per-method comparison (n={len(frame)}, mean answer time {avg_time:.1f}s) ===")
        print(format_mode_table(summary, include_sim=True))
        return

    # ── Interactive multi-turn mode ──────────────────────────────────────────
    if args.interactive:
        print("Multi-turn conversation mode. Type 'quit' to exit, 'reset' to clear history.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.lower() == "quit":
                break
            if query.lower() == "reset":
                rag.reset_conversation()
                print("Conversation history cleared.\n")
                continue

            result = rag.ask(query)
            if args.show_context:
                print("\n--- Context ---")
                print(result["context"])
                print("---\n")
            print(f"\nPage:   {result['page']}")
            print(f"Answer: {result['answer']}\n")
        return

    # ── Single query mode ────────────────────────────────────────────────────
    query = args.query or DEFAULT_QUERY
    if not query:
        raise ValueError("Provide --query, --interactive, or --qa-csv.")

    result = rag.ask(query)

    if args.show_context:
        print(f"Query:           {result['query']}")
        print(f"Rewritten query: {result['rewritten_query']}")
        print(f"BM25 rank:       {result['retrieval']['bm25_rank']}")
        print(f"Vector rank:     {result['retrieval']['vector_rank']}")
        print(f"Fused rank:      {result['retrieval']['fused_rank']}")
        print("\n=== Retrieved Context ===")
        print(result["context"])

    print(f"\nPage:   {result['page']}")
    print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    main()
