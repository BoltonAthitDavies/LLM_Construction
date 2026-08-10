"""LLM-as-a-judge: score RAG answers against ground truth on the POLRAG rubric.

An LLM judge (Gemini) grades each generated answer on two content dimensions:
  - Engineering Answer Correctness (0-40)
  - Document Grounding Accuracy   (0-30)

It reads a results CSV produced by ``src/RAG.py`` (columns ``Question``,
``Answer`` = ground truth, ``predicted_<mode>_Answer``, ``Page``,
``predicted_<mode>_Page``) and writes the judge scores back as new columns
``judge_<mode>_eng``, ``judge_<mode>_grounding``, ``judge_<mode>_note``.

Usage:
    python src/llm_as_a_judge.py --input-csv result/testset_lvl1_large.csv \
        --output-csv result/testset_lvl1_large_judged.csv --mode fused
    python src/llm_as_a_judge.py --input-csv result/testset_lvl2_small.csv --mode fused --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def load_api_keys(cli_key: str | None = None) -> list[str]:
    """Resolve Gemini keys: geminikey.txt next to this file, then project root, then env."""
    here = Path(__file__).parent
    for path in (here / "geminikey.txt", here.parent / "geminikey.txt"):
        if path.exists():
            keys = [k.strip() for k in path.read_text(encoding="utf-8").splitlines() if k.strip()]
            if keys:
                return keys
    env = cli_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return [env] if env else []


JUDGE_PROMPT = """You are an expert civil-engineering examiner. Grade an AI assistant's answer about \
tunnel-segment repair against an expert reference answer. Score TWO dimensions; apply the bands strictly.

[Engineering Answer Correctness]  integer 0-40
  36-40 Excellent: identifies the correct repair method/criteria matching the reference, WITH the complete
        numeric conditions (crack width, depth, void size, mixing ratio, etc.).
  26-35 Good: method/answer correct but some numeric detail is missing.
  16-25 Fair: partially correct; departs from the reference in method or key values.
  0-15  Poor: wrong, contradicts the reference, or fails to answer (e.g. abstains when the reference has an answer).

[Document Grounding Accuracy]  integer 0-30
  26-30 Excellent: fully supported by the source with no hallucination; the cited page matches the reference page.
  18-25 Good: largely faithful, minor over-summarisation or noisy/extra citations.
  0-17  Poor: clear hallucination or unsupported claim, or the cited page does not match the reference page.

Judge meaning, not surface wording; the answer and reference may be in Thai or English. Be strict and consistent.

QUESTION:
{q}

REFERENCE ANSWER (ground truth):
{gt}
REFERENCE PAGE(S): {gtp}

ASSISTANT ANSWER TO GRADE:
{pred}
CITED PAGE(S): {predp}

Respond with ONLY a JSON object:
{{"engineering_correctness": <int 0-40>, "grounding": <int 0-30>, "justification": "<=20 words"}}"""


class GeminiJudge:
    FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-flash-latest"]
    _RATE = ("RESOURCE_EXHAUSTED", "RATE_LIMIT", "429", "quota")

    def __init__(self, keys: list[str], model: str = "gemini-2.5-flash") -> None:
        if genai is None:
            raise ImportError("Missing 'google-genai'. Install: pip install google-genai")
        if not keys:
            raise ValueError("No Gemini API key found (src/geminikey.txt or GOOGLE_API_KEY).")
        self.keys = keys
        self.model = model
        self._i = 0
        self.client = genai.Client(api_key=keys[0])
        self._cfg = None
        if types is not None:
            try:
                self._cfg = types.GenerateContentConfig(
                    temperature=0.0, response_mime_type="application/json"
                )
            except Exception:
                self._cfg = None

    def _rotate(self) -> None:
        self._i = (self._i + 1) % len(self.keys)
        self.client = genai.Client(api_key=self.keys[self._i])

    def _generate(self, prompt: str) -> str:
        models = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        last = ""
        for model_name in models:
            for _ in range(len(self.keys)):
                try:
                    if self._cfg is not None:
                        resp = self.client.models.generate_content(
                            model=model_name, contents=prompt, config=self._cfg)
                    else:
                        resp = self.client.models.generate_content(model=model_name, contents=prompt)
                    return getattr(resp, "text", "") or ""
                except Exception as exc:  # noqa: BLE001
                    last = str(exc)
                    if any(sig in last for sig in self._RATE):
                        self._rotate()
                        time.sleep(1.0)
                    else:
                        break
        raise RuntimeError(last or "judge generation failed")

    def score(self, q: str, gt: str, gtp: str, pred: str, predp: str) -> tuple[int | None, int | None, str]:
        raw = self._generate(JUDGE_PROMPT.format(q=q, gt=gt, gtp=gtp, pred=pred, predp=predp))
        return parse_scores(raw)


def parse_scores(text: str) -> tuple[int | None, int | None, str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            eng = int(round(float(data.get("engineering_correctness", data.get("eng", -1)))))
            grnd = int(round(float(data.get("grounding", data.get("document_grounding", -1)))))
            note = str(data.get("justification", data.get("note", "")))[:200]
            eng = max(0, min(40, eng)) if eng >= 0 else None
            grnd = max(0, min(30, grnd)) if grnd >= 0 else None
            return eng, grnd, note
        except Exception:
            pass
    return None, None, f"parse-error: {text[:80]}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-a-judge rubric scorer (Gemini).")
    p.add_argument("--input-csv", required=True, help="Results CSV from RAG.py.")
    p.add_argument("--output-csv", default=None, help="Output path (default: <input>_judged.csv).")
    p.add_argument("--mode", default="fused", help="Which predicted_<mode>_Answer to grade.")
    p.add_argument("--question-col", default="Question")
    p.add_argument("--gt-col", default="Answer", help="Ground-truth answer column.")
    p.add_argument("--gt-page-col", default="Page")
    p.add_argument("--judge-model", default="gemini-2.5-flash")
    p.add_argument("--gemini-api-key", default=None)
    p.add_argument("--limit", type=int, default=None, help="Score only the first N rows.")
    return p.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    judge = GeminiJudge(load_api_keys(args.gemini_api_key), args.judge_model)

    in_path = Path(args.input_csv)
    frame = pd.read_csv(in_path)
    if args.limit is not None:
        frame = frame.head(args.limit).copy()

    mode = args.mode
    ans_col = f"predicted_{mode}_Answer"
    page_col = f"predicted_{mode}_Page"
    if ans_col not in frame.columns:
        raise ValueError(f"Column '{ans_col}' not found. The CSV needs predicted_<mode>_Answer.")

    engs: list = []
    grnds: list = []
    notes: list = []
    t0 = time.time()
    for idx, row in frame.iterrows():
        question = str(row[args.question_col]).strip()
        if not question or question.lower() == "nan":
            engs.append("")
            grnds.append("")
            notes.append("")
            continue
        gt = str(row.get(args.gt_col, ""))
        gtp = str(row.get(args.gt_page_col, ""))
        pred = str(row.get(ans_col, ""))
        predp = str(row.get(page_col, ""))
        try:
            eng, grnd, note = judge.score(question, gt, gtp, pred, predp)
        except Exception as exc:  # noqa: BLE001
            eng, grnd, note = None, None, f"error: {str(exc)[:80]}"
        engs.append(eng)
        grnds.append(grnd)
        notes.append(note)
        print(f"  judged {idx + 1}/{len(frame)} - eng={eng} grounding={grnd}")

    frame[f"judge_{mode}_eng"] = engs
    frame[f"judge_{mode}_grounding"] = grnds
    frame[f"judge_{mode}_note"] = notes

    out_path = Path(args.output_csv) if args.output_csv else in_path.with_name(
        f"{in_path.stem}_judged{in_path.suffix}")
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")

    e = pd.to_numeric(frame[f"judge_{mode}_eng"], errors="coerce").dropna()
    g = pd.to_numeric(frame[f"judge_{mode}_grounding"], errors="coerce").dropna()
    print(
        f"\nJudge '{args.judge_model}' on {mode} - n={len(e)} | "
        f"Eng {e.mean():.1f}/40 | Grounding {g.mean():.1f}/30 | "
        f"({time.time() - t0:.0f}s)"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
