"""Evaluation harness for the Agrivoltaics AI Assistant RAG pipeline.

Measures, over a curated question set:
  1. Retrieval hit rate - did the expected source appear in the retrieved chunks?
  2. Groundedness - does the generated answer stay supported by retrieved context
     (scored 1-5 by an LLM judge)?
  3. Grounded vs. ungrounded delta - groundedness of the RAG answer minus the
     groundedness of the same model answering with no retrieved context.

Outputs a Markdown report and a CSV of per-question results.

Usage (from project root, with the venv active):
    python eval/run_eval.py
    python eval/run_eval.py --limit 4 --no-judge
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from config.settings import SYSTEM_PROMPT, Settings  # noqa: E402
from src.rag.retriever import ContextRetriever  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("agrivoltaics.eval")

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "eval_questions.jsonl"


@dataclass
class QuestionResult:
    """Per-question evaluation record."""

    qid: str
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str] = field(default_factory=list)
    top_distance: float | None = None
    retrieval_hit: bool | None = None
    grounded_answer: str = ""
    ungrounded_answer: str = ""
    grounded_score: int | None = None
    ungrounded_score: int | None = None


def load_questions(limit: int | None = None) -> list[dict]:
    """Read the JSONL question set."""
    questions: list[dict] = []
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit is not None:
        questions = questions[:limit]
    return questions


def generate_answer(client: OpenAI, model: str, system_content: str, question: str) -> str:
    """Non-streaming completion for evaluation."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Answer generation failed: %s", exc)
        return f"[generation error: {exc}]"


def judge_groundedness(client: OpenAI, model: str, context: str, answer: str) -> int | None:
    """LLM-as-judge: rate 1-5 how well the answer is supported by the context."""
    if not context.strip():
        return None
    judge_prompt = (
        "You are a strict evaluator. Given REFERENCE CONTEXT and an ANSWER, rate how well "
        "the answer's factual claims are supported by the context on a scale of 1 to 5:\n"
        "5 = every specific claim is supported by the context\n"
        "3 = partially supported; some claims not in context\n"
        "1 = mostly unsupported or contradicts the context\n"
        "Reply with ONLY the integer.\n\n"
        f"REFERENCE CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"[1-5]", text)
        return int(match.group(0)) if match else None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Judge failed: %s", exc)
        return None


def run(limit: int | None, use_judge: bool) -> None:
    """Execute the evaluation and write reports."""
    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)
    retriever = ContextRetriever(settings, client)
    model = settings.chat_model

    questions = load_questions(limit)
    results: list[QuestionResult] = []

    for item in questions:
        qid = item["id"]
        question = item["question"]
        expected = item.get("expected_sources", []) or []
        print(f"Evaluating {qid}: {question}")

        retrieval = retriever.retrieve(question)
        retrieved_sources = [c.source for c in retrieval.chunks]
        top_distance = retrieval.chunks[0].score if retrieval.chunks else None
        context = retrieval.formatted_context() if retrieval.chunks else ""

        hit: bool | None
        if expected:
            hit = any(src in retrieved_sources for src in expected)
        else:
            hit = None

        grounded_system = SYSTEM_PROMPT
        if context:
            grounded_system += (
                "\n\n[Private background notes - use to inform your answer]\n" + context
            )
        ungrounded_system = SYSTEM_PROMPT

        grounded_answer = generate_answer(client, model, grounded_system, question)
        ungrounded_answer = generate_answer(client, model, ungrounded_system, question)

        grounded_score = None
        ungrounded_score = None
        if use_judge and context:
            grounded_score = judge_groundedness(client, model, context, grounded_answer)
            ungrounded_score = judge_groundedness(client, model, context, ungrounded_answer)

        results.append(
            QuestionResult(
                qid=qid,
                question=question,
                expected_sources=expected,
                retrieved_sources=retrieved_sources,
                top_distance=top_distance,
                retrieval_hit=hit,
                grounded_answer=grounded_answer,
                ungrounded_answer=ungrounded_answer,
                grounded_score=grounded_score,
                ungrounded_score=ungrounded_score,
            )
        )

    _write_reports(settings, results, use_judge)


def _write_reports(settings: Settings, results: list[QuestionResult], use_judge: bool) -> None:
    """Persist CSV + Markdown report and print a summary."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    scored_hits = [r for r in results if r.retrieval_hit is not None]
    hit_rate = (
        sum(1 for r in scored_hits if r.retrieval_hit) / len(scored_hits)
        if scored_hits
        else None
    )
    grounded_scores = [r.grounded_score for r in results if r.grounded_score is not None]
    ungrounded_scores = [r.ungrounded_score for r in results if r.ungrounded_score is not None]
    avg_grounded = sum(grounded_scores) / len(grounded_scores) if grounded_scores else None
    avg_ungrounded = (
        sum(ungrounded_scores) / len(ungrounded_scores) if ungrounded_scores else None
    )
    delta = (
        avg_grounded - avg_ungrounded
        if avg_grounded is not None and avg_ungrounded is not None
        else None
    )

    csv_path = EVAL_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "question",
                "expected_sources",
                "retrieved_sources",
                "top_distance",
                "retrieval_hit",
                "grounded_score",
                "ungrounded_score",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.qid,
                    r.question,
                    "; ".join(r.expected_sources),
                    "; ".join(r.retrieved_sources),
                    f"{r.top_distance:.4f}" if r.top_distance is not None else "",
                    "" if r.retrieval_hit is None else r.retrieval_hit,
                    "" if r.grounded_score is None else r.grounded_score,
                    "" if r.ungrounded_score is None else r.ungrounded_score,
                ]
            )

    def fmt(value: float | None, pattern: str = "{:.2f}") -> str:
        return pattern.format(value) if value is not None else "n/a"

    lines: list[str] = []
    lines.append("# Agrivoltaics AI Assistant - Evaluation Report")
    lines.append("")
    lines.append(f"Run: {timestamp}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Chat model: `{settings.chat_model}`")
    lines.append(f"- Embedding model: `{settings.embedding_model}`")
    lines.append(f"- top_k: {settings.retrieval_top_k}")
    lines.append(f"- Distance threshold: {settings.retrieval_distance_threshold}")
    lines.append(f"- Candidate multiplier: {settings.retrieval_candidate_multiplier}")
    lines.append(f"- Chunk size / overlap: {settings.chunk_size} / {settings.chunk_overlap}")
    lines.append(f"- LLM judge enabled: {use_judge}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Questions evaluated: {len(results)}")
    lines.append(
        f"- Retrieval hit rate (questions with expected sources): {fmt(hit_rate, '{:.0%}')}"
        f" ({len(scored_hits)} questions)"
    )
    lines.append(f"- Avg groundedness (RAG on): {fmt(avg_grounded)} / 5")
    lines.append(f"- Avg groundedness (RAG off): {fmt(avg_ungrounded)} / 5")
    lines.append(f"- Grounded vs ungrounded delta: {fmt(delta, '{:+.2f}')}")
    lines.append("")
    lines.append("## Per-question results")
    lines.append("")
    lines.append("| ID | Hit | Top dist | Grounded | Ungrounded | Question |")
    lines.append("|----|-----|----------|----------|------------|----------|")
    for r in results:
        hit_label = "-" if r.retrieval_hit is None else ("yes" if r.retrieval_hit else "no")
        dist = f"{r.top_distance:.3f}" if r.top_distance is not None else "-"
        gs = "-" if r.grounded_score is None else str(r.grounded_score)
        us = "-" if r.ungrounded_score is None else str(r.ungrounded_score)
        lines.append(f"| {r.qid} | {hit_label} | {dist} | {gs} | {us} | {r.question} |")
    lines.append("")

    report_path = EVAL_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Questions evaluated: {len(results)}")
    print(f"Retrieval hit rate: {fmt(hit_rate, '{:.0%}')} ({len(scored_hits)} questions)")
    print(f"Avg groundedness (RAG on):  {fmt(avg_grounded)} / 5")
    print(f"Avg groundedness (RAG off): {fmt(avg_ungrounded)} / 5")
    print(f"Delta (on - off): {fmt(delta, '{:+.2f}')}")
    print(f"\nReport: {report_path}")
    print(f"CSV:    {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge groundedness scoring (faster, cheaper).",
    )
    args = parser.parse_args()
    run(limit=args.limit, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
