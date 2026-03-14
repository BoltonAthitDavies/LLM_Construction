"""RAG pipeline matching the provided flow image.

Flow:
1) Keyword Retrieval (ONER-style token keyword matching)
2) Full-Text Search (Thai + English)
3) Vector Search (cosine similarity)
4) Reciprocal Rank Fusion (RRF)
5) Answer synthesis (multilingual)

Usage:
    python src/RAG.py --docs dataset_json --query "How to repair blowholes?"
    python src/RAG.py --docs dataset_json --query "วิธีซ่อมโพรงอากาศคืออะไร"

Required environment variable:
    OPENAI_API_KEY (for embeddings)
    GOOGLE_API_KEY or GEMINI_API_KEY (for Gemini answer generation)
"""

# python src/RAG.py --docs dataset_json --query "วิธีซ่อมโพรกาศคืออะไร"
# python src/RAG.py --docs dataset_json --qa-csv tablev2/QA_test.csv --output-csv tablev2/QA_test_predicted.csv
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

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


# If you already have your query, you can place it here and run without --query.
DEFAULT_QUERY = ""


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings for environments without OpenAI key."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+|[\u0E00-\u0E7F]+", text.lower())

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
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


class RAGFlow:
    def __init__(
        self,
        documents_dir: str,
        extensions: Iterable[str] = (".txt", ".md", ".json", ".csv"),
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        gemini_model: str = "gemini-3-pro-preview",
        gemini_api_key: str | None = None,
        top_k_each: int = 8,
        top_k_fused: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.gemini_model = gemini_model
        self.top_k_each = top_k_each
        self.top_k_fused = top_k_fused
        self.rrf_k = rrf_k

        if genai is None:
            raise ImportError(
                "Missing package 'google-genai'. Install it with: pip install google-genai"
            )

        key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Missing Gemini API key. Set GOOGLE_API_KEY/GEMINI_API_KEY or pass --gemini-api-key."
            )
        self.client = genai.Client(api_key=key)

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.embeddings: Embeddings = OpenAIEmbeddings(api_key=openai_key)
        else:
            self.embeddings = LocalHashEmbeddings(dim=512)

        self.documents = self._load_and_chunk_documents()
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.doc_tokens = [self._tokenize(doc.page_content) for doc in self.documents]
        self.idf = self._build_idf(self.doc_tokens)

    def _iter_source_files(self) -> list[Path]:
        if not self.documents_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.documents_dir}")

        files = [
            p
            for p in self.documents_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.extensions
        ]
        if not files:
            raise ValueError(
                f"No files found in {self.documents_dir} with extensions {self.extensions}"
            )
        return files

    def _extract_text_and_assets(self, file_path: Path) -> tuple[str, list[str], list[str]]:
        images: list[str] = []
        tables: list[str] = []

        if file_path.suffix.lower() != ".json":
            return file_path.read_text(encoding="utf-8", errors="ignore"), images, tables

        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw, images, tables

        text_parts: list[str] = []

        def walk(value: Any, parent_key: str = "") -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(v, str(k).lower())
                return

            if isinstance(value, list):
                for item in value:
                    walk(item, parent_key)
                return

            if isinstance(value, (str, int, float, bool)):
                text_value = str(value)
                text_parts.append(text_value)
                if parent_key and any(tag in parent_key for tag in ("image", "figure", "img", "png", "jpg")):
                    images.append(text_value)
                if parent_key and "table" in parent_key:
                    tables.append(text_value)

        walk(payload)
        return "\n".join(text_parts), images, tables

    def _load_and_chunk_documents(self) -> list[Document]:
        source_files = self._iter_source_files()

        base_docs: list[Document] = []
        for file_path in source_files:
            text, images, tables = self._extract_text_and_assets(file_path)
            if not text.strip():
                continue
            base_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "images": images[:20],
                        "tables": tables[:20],
                    },
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(base_docs)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+|[\u0E00-\u0E7F]+", text.lower())

    def _build_idf(self, tokenized_docs: list[list[str]]) -> dict[str, float]:
        n_docs = len(tokenized_docs)
        df: dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        idf: dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log((n_docs + 1) / (freq + 1)) + 1.0
        return idf

    def _keyword_retrieval(self, query: str, top_k: int) -> list[int]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            score = 0.0
            for qt in q_tokens:
                score += tf.get(qt, 0) * self.idf.get(qt, 0.0)

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]

    def _full_text_retrieval(self, query: str, top_k: int) -> list[int]:
        q = query.strip().lower()
        q_tokens = self._tokenize(query)
        if not q:
            return []

        scores: list[tuple[int, float]] = []
        for idx, doc in enumerate(self.documents):
            text = doc.page_content.lower()
            phrase_hits = text.count(q)
            token_hits = 0
            for token in q_tokens:
                token_hits += text.count(token)

            score = phrase_hits * 5.0 + token_hits
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]

    def _vector_retrieval(self, query: str, top_k: int) -> list[int]:
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        index_by_key = {self._doc_key(doc): i for i, doc in enumerate(self.documents)}

        ranked: list[tuple[int, float]] = []
        for doc, distance in results:
            key = self._doc_key(doc)
            idx = index_by_key.get(key)
            if idx is None:
                continue
            ranked.append((idx, float(distance)))

        ranked.sort(key=lambda x: x[1])
        return [idx for idx, _ in ranked]

    @staticmethod
    def _doc_key(doc: Document) -> tuple[str, str]:
        return (str(doc.metadata.get("source", "")), doc.page_content)

    def _rrf_fusion(self, rankings: list[list[int]], top_k: int) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_idx in enumerate(ranking, start=1):
                fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in ordered[:top_k]]

    def _build_context(self, doc_ids: list[int]) -> tuple[str, list[str], list[str]]:
        parts: list[str] = []
        image_refs: list[str] = []
        table_refs: list[str] = []

        for i, doc_idx in enumerate(doc_ids, start=1):
            doc = self.documents[doc_idx]
            source = str(doc.metadata.get("source", "unknown"))
            images = doc.metadata.get("images", [])
            tables = doc.metadata.get("tables", [])

            for item in images:
                image_refs.append(str(item))
            for item in tables:
                table_refs.append(str(item))

            parts.append(f"[Chunk {i}] Source: {source}\n{doc.page_content}")

        context_text = "\n\n".join(parts)
        return context_text, image_refs[:20], table_refs[:20]

    def _extractive_fallback_answer(self, query: str, context: str, max_sentences: int = 5) -> str:
        q_tokens = set(self._tokenize(query))
        if not context.strip():
            return "No context available to answer this question."

        chunks = re.split(r"(?<=[.!?\n])\s+", context)
        scored: list[tuple[float, str]] = []
        for sent in chunks:
            clean = sent.strip()
            if len(clean) < 20:
                continue
            s_tokens = self._tokenize(clean)
            if not s_tokens:
                continue
            overlap = sum(1 for t in s_tokens if t in q_tokens)
            score = overlap / max(1, len(set(s_tokens)))
            if overlap > 0:
                scored.append((score, clean))

        if not scored:
            preview = context[:900].strip()
            return (
                "Gemini quota is currently exhausted. Returning retrieved context summary instead:\n\n"
                f"{preview}"
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [s for _, s in scored[:max_sentences]]
        return (
            "Gemini quota is currently exhausted. Returning best extractive answer from retrieved context:\n\n"
            + "\n".join(selected)
        )

    def _generate_with_gemini(self, prompt: str) -> tuple[str, str]:
        candidate_models = [
            self.gemini_model,
            "gemini-2.5-flash",
            "gemini-1.5-flash",
        ]
        seen: set[str] = set()
        model_list: list[str] = []
        for model_name in candidate_models:
            if model_name and model_name not in seen:
                seen.add(model_name)
                model_list.append(model_name)

        last_error = ""
        for model_name in model_list:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if text:
                    return text.strip(), model_name
                return str(response), model_name
            except Exception as exc:
                last_error = str(exc)
                # Brief backoff for transient API issues.
                time.sleep(1.0)

        raise RuntimeError(last_error or "Gemini generation failed")

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
            match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        answer_value = capture(r"Answer\s*:\s*(.*?)(?:\n\s*Page\s*:|\Z)")
        return {
            "Rainbow Group": capture(r"Rainbow Group\s*:\s*(.+)") or capture(r"Rainbow_Group\s*:\s*(.+)"),
            "Category": capture(r"Category\s*:\s*(.+)"),
            "Answer": answer_value,
            "Page": capture(r"Page\s*:\s*(.+)") or capture(r"Reference Page\s*:\s*(.+)"),
        }

    @staticmethod
    def _extract_page_from_source(source: str) -> str:
        match = re.search(r"page[_-]?(\d+)", source, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _infer_category(question: str, answer: str) -> str:
        text = f"{question} {answer}".lower()
        if any(token in text for token in ["risk", "safety", "อันตราย", "ปฐมพยาบาล", "chemical", "สารเคมี"]):
            return "Human Risk"
        if any(token in text for token in ["environment", "preventive", "สิ่งแวดล้อม", "คุณภาพน้ำ", "คุณภาพดิน", "ดับเพลิง"]):
            return "Environmental / Preventive Measures"
        if any(token in text for token in ["material", "mixing ratio", "potlife", "compressive strength", "วัสดุ", "อัตราส่วนผสม", "กำลังอัด"]):
            return "Material"
        if any(token in text for token in ["repair", "crack", "spalling", "blowholes", "voids", "ซ่อม", "รอยร้าว", "แตกบิ่น", "โพรง", "ฟองอากาศ"]):
            return "Repair Method"
        return "General / Admin"

    @staticmethod
    def _infer_rainbow_group(category: str) -> str:
        if category in {
            "Repair Method",
            "Material",
            "Human Risk",
            "Environmental / Preventive Measures",
        }:
            return "Construction-Related"
        return "Non-Construction Questions"

    @staticmethod
    def _normalize_prediction_payload(
        payload: dict[str, Any],
        raw_answer: str,
        question: str,
        fallback_page: str,
    ) -> dict[str, str]:
        def get_value(*keys: str) -> str:
            for key in keys:
                if key in payload and payload[key] is not None:
                    return str(payload[key]).strip()
            return ""

        category = get_value("Category", "category")
        if not category:
            category = RAGFlow._infer_category(question, raw_answer)

        rainbow_group = get_value("Rainbow Group", "rainbow_group")
        if not rainbow_group:
            rainbow_group = RAGFlow._infer_rainbow_group(category)

        answer = get_value("Answer", "answer") or raw_answer.strip() or "unknown"
        page = get_value("Page", "page", "Reference Page", "reference_page") or fallback_page or "unknown"

        return {
            "predicted_Rainbow Group": rainbow_group,
            "predicted_Category": category,
            "predicted_Answer": answer,
            "predicted_Page": page,
        }

    def predict_expected_columns(self, question: str) -> dict[str, Any]:
        keyword_rank = self._keyword_retrieval(question, top_k=self.top_k_each)
        fulltext_rank = self._full_text_retrieval(question, top_k=self.top_k_each)
        vector_rank = self._vector_retrieval(question, top_k=self.top_k_each)

        fused_ids = self._rrf_fusion(
            [keyword_rank, fulltext_rank, vector_rank],
            top_k=self.top_k_fused,
        )
        context, images, tables = self._build_context(fused_ids)

        prompt = (
            "You are an expert civil engineering QA assistant specialized in the document 'Method for Repairing Tunnel Segment'.\n"
            "Use only the retrieved context.\n"
            "Classify and answer the question using this schema exactly.\n"
            "Return either valid JSON or these exact labeled lines:\n"
            "Rainbow Group: ...\n"
            "Category: ...\n"
            "Answer: ...\n"
            "Page: ...\n"
            "Rainbow Group must be either 'Construction-Related' or 'Non-Construction Questions'.\n"
            "Category should match the document taxonomy when possible.\n"
            "Page should be the exact reference page number if available, else empty string.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context}\n"
        )

        answer_model = ""
        raw_answer = ""
        parsed: dict[str, Any] | None = None
        try:
            raw_answer, answer_model = self._generate_with_gemini(prompt)
            parsed = self._extract_json_object(raw_answer)
            if parsed is None:
                parsed = self._extract_labeled_fields(raw_answer)
        except Exception as exc:
            err_text = str(exc)
            if "RESOURCE_EXHAUSTED" in err_text or "RATE_LIMIT_EXCEEDED" in err_text or "429" in err_text:
                raw_answer = self._extractive_fallback_answer(question, context)
                answer_model = "extractive-fallback"
            else:
                raise

        fallback_page = ""
        for doc_idx in fused_ids:
            fallback_page = self._extract_page_from_source(str(self.documents[doc_idx].metadata.get("source", "")))
            if fallback_page:
                break

        predictions = self._normalize_prediction_payload(
            parsed or {},
            raw_answer,
            question,
            fallback_page,
        )
        return {
            "question": question,
            "answer_model": answer_model,
            "raw_answer": raw_answer,
            "predictions": predictions,
            "query_results": {
                "text_context": context,
                "images": images,
                "tables": tables,
            },
        }

    def ask(self, query: str) -> dict[str, Any]:
        keyword_rank = self._keyword_retrieval(query, top_k=self.top_k_each)
        fulltext_rank = self._full_text_retrieval(query, top_k=self.top_k_each)
        vector_rank = self._vector_retrieval(query, top_k=self.top_k_each)

        fused_ids = self._rrf_fusion(
            [keyword_rank, fulltext_rank, vector_rank],
            top_k=self.top_k_fused,
        )

        context, images, tables = self._build_context(fused_ids)

        prompt = (
            "You are a multilingual QA assistant. "
            "Answer using only the retrieved context. "
            "If context is insufficient, state uncertainty clearly.\n\n"
            f"Question: {query}\n\n"
            "Retrieved context:\n"
            f"{context}\n\n"
            "Provide a concise and factual answer in the same language as the question."
        )

        llm_model_used = ""
        try:
            raw_answer, llm_model_used = self._generate_with_gemini(prompt)
        except Exception as exc:
            err_text = str(exc)
            if "RESOURCE_EXHAUSTED" in err_text or "RATE_LIMIT_EXCEEDED" in err_text or "429" in err_text:
                raw_answer = self._extractive_fallback_answer(query=query, context=context)
                llm_model_used = "extractive-fallback"
            else:
                raise

        return {
            "query": query,
            "retrieval": {
                "keyword_rank": keyword_rank,
                "fulltext_rank": fulltext_rank,
                "vector_rank": vector_rank,
                "fused_rank": fused_ids,
            },
            "query_results": {
                "text_context": context,
                "images": images,
                "tables": tables,
            },
            "answer_model": llm_model_used,
            "raw_answer": raw_answer,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG flow with RRF fusion.")
    parser.add_argument("--docs", required=True, help="Directory containing source docs.")
    parser.add_argument("--query", default=None, help="User question.")
    parser.add_argument("--qa-csv", default=None, help="CSV file containing Question and expected columns.")
    parser.add_argument("--output-csv", default=None, help="Output CSV path for batch predictions.")
    parser.add_argument("--question-col", default="Question", help="Question column name in --qa-csv.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt", ".md", ".json", ".csv"],
        help="Extensions to index.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--gemini-model", default="gemini-3-pro-preview")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--top-k-each", type=int, default=8)
    parser.add_argument("--top-k-fused", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Print query results (text/images/tables) before final answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rag = RAGFlow(
        documents_dir=args.docs,
        extensions=args.extensions,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        gemini_model=args.gemini_model,
        gemini_api_key=args.gemini_api_key,
        top_k_each=args.top_k_each,
        top_k_fused=args.top_k_fused,
        rrf_k=args.rrf_k,
    )

    if args.qa_csv:
        qa_csv_path = Path(args.qa_csv)
        if not qa_csv_path.exists():
            raise FileNotFoundError(f"QA CSV not found: {qa_csv_path}")

        frame = pd.read_csv(qa_csv_path)
        if args.question_col not in frame.columns:
            raise ValueError(
                f"Question column '{args.question_col}' not found in {qa_csv_path}. Columns: {list(frame.columns)}"
            )

        predicted_rainbow_group: list[str] = []
        predicted_category: list[str] = []
        predicted_answer: list[str] = []
        predicted_page: list[str] = []
        predicted_model: list[str] = []
        predicted_raw_answer: list[str] = []

        for idx, row in frame.iterrows():
            question = str(row[args.question_col]).strip()
            if not question:
                predicted_rainbow_group.append("")
                predicted_category.append("")
                predicted_answer.append("")
                predicted_page.append("")
                predicted_model.append("")
                predicted_raw_answer.append("")
                continue

            prediction = rag.predict_expected_columns(question)
            predicted_rainbow_group.append(prediction["predictions"]["predicted_Rainbow Group"])
            predicted_category.append(prediction["predictions"]["predicted_Category"])
            predicted_answer.append(prediction["predictions"]["predicted_Answer"])
            predicted_page.append(prediction["predictions"]["predicted_Page"])
            predicted_model.append(prediction["answer_model"])
            predicted_raw_answer.append(prediction["raw_answer"])
            print(f"Processed row {idx + 1}/{len(frame)}")

        frame["predicted_Rainbow Group"] = predicted_rainbow_group
        frame["predicted_Category"] = predicted_category
        frame["predicted_Answer"] = predicted_answer
        frame["predicted_Page"] = predicted_page
        frame["predicted_answer_model"] = predicted_model
        frame["predicted_raw_answer"] = predicted_raw_answer

        output_csv = Path(args.output_csv) if args.output_csv else qa_csv_path.with_name(
            f"{qa_csv_path.stem}_predicted{qa_csv_path.suffix}"
        )
        frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"Saved predictions to: {output_csv}")
        return

    query = args.query or DEFAULT_QUERY
    if not query:
        raise ValueError("Provide --query or set DEFAULT_QUERY in src/RAG.py, or use --qa-csv")

    result = rag.ask(query)

    print("\n=== Retrieval (RRF) ===")
    print("Keyword:", result["retrieval"]["keyword_rank"])
    print("Full-text:", result["retrieval"]["fulltext_rank"])
    print("Vector:", result["retrieval"]["vector_rank"])
    print("Fused:", result["retrieval"]["fused_rank"])

    if args.show_results:
        print("\n=== Query Results: text + images + tables ===")
        print(result["query_results"]["text_context"])
        print("\nImages:", result["query_results"]["images"])
        print("Tables:", result["query_results"]["tables"])

    print("\n=== Raw Answer (multilingual) ===\n")
    print("Answer model:", result.get("answer_model", "unknown"))
    print(result["raw_answer"])


if __name__ == "__main__":
    main()
