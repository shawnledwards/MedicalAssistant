#!/usr/bin/env python3
"""
Evaluation script for Medical Assistant RAG.

Runs test questions through all three personas and scores each answer for:
  - Groundedness: does the answer stay within the retrieved context?
  - Relevance:    does the answer address the question asked?

Both are scored 1-5 by the same Mistral model (LLM-as-judge).
Note: scores may be slightly generous since the model evaluates its own outputs.

Usage:
    conda run -n medassist python eval.py
    conda run -n medassist python eval.py --personas scientist doctor
    conda run -n medassist python eval.py --no-ratings   # skip scoring, faster
"""

import argparse
import sys
from datetime import datetime

from tabulate import tabulate

from medical_assistant.config.personas import PERSONAS
from medical_assistant.config.settings import Settings
from medical_assistant.evaluation.raters import rate_groundedness, rate_relevance
from medical_assistant.rag.pipeline import RAGPipeline

TEST_QUESTIONS = [
    # Original notebook test queries
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the necessary precautions and treatment steps for a person who has "
    "fractured their leg during a hiking trip, and what should be considered for "
    "their care and recovery?",
    # Additional coverage queries
    "What are the symptoms and treatment options for type 2 diabetes?",
    "How is hypertension diagnosed and managed?",
    "What are the signs of a stroke and what immediate actions should be taken?",
    # Out-of-scope query — should trigger the no-context fallback
    "What is the recommended dosage of ibuprofen for a 6-year-old child?",
]


def parse_score(rating_result: str) -> str:
    """Extract the numeric score from the rater's free-text output."""
    for token in rating_result.split():
        token = token.strip(".,:()/")
        if token.isdigit() and 1 <= int(token) <= 5:
            return token
    return "?"


def run_eval(personas: list[str], include_ratings: bool) -> None:
    settings = Settings()
    pipeline = RAGPipeline(settings=settings)

    print(f"\nMedical Assistant RAG — Evaluation Report")
    print(f"{'=' * 60}")
    print(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Personas  : {', '.join(personas)}")
    print(f"Questions : {len(TEST_QUESTIONS)}")
    print(f"Ratings   : {'yes (LLM-as-judge)' if include_ratings else 'no'}")
    print()

    summary_rows = []

    for q_idx, question in enumerate(TEST_QUESTIONS, 1):
        short_q = question[:70] + ("…" if len(question) > 70 else "")
        print(f"Q{q_idx}: {short_q}")
        print("-" * 60)

        for persona in personas:
            result = pipeline.run_query(question=question, persona=persona)
            answer = result["answer"]
            context = result.get("context", "")

            print(f"  [{persona.upper()}]")
            print(f"  {answer[:300]}{'…' if len(answer) > 300 else ''}")

            g_score = r_score = "—"
            g_note = r_note = ""

            if include_ratings and context:
                g = rate_groundedness(context=context, answer=answer, settings=settings)
                r = rate_relevance(question=question, answer=answer, settings=settings)
                g_score = parse_score(g["result"])
                r_score = parse_score(r["result"])
                g_note = g["result"].split("\n")[0][:80]
                r_note = r["result"].split("\n")[0][:80]
                print(f"  Groundedness {g_score}/5 — {g_note}")
                print(f"  Relevance    {r_score}/5 — {r_note}")

            summary_rows.append([f"Q{q_idx}", persona, g_score, r_score])
            print()

        print()

    if include_ratings:
        print("Summary")
        print("=" * 60)
        print(tabulate(
            summary_rows,
            headers=["Question", "Persona", "Grounded", "Relevant"],
            tablefmt="simple",
        ))
        print()
        print("Scores are 1–5 (5 = best). '—' = no context retrieved (fallback answer).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Medical Assistant RAG personas")
    parser.add_argument(
        "--personas",
        nargs="+",
        choices=list(PERSONAS.keys()),
        default=list(PERSONAS.keys()),
        help="Personas to evaluate (default: all)",
    )
    parser.add_argument(
        "--no-ratings",
        action="store_true",
        help="Skip LLM scoring — just print answers (much faster)",
    )
    args = parser.parse_args()
    run_eval(personas=args.personas, include_ratings=not args.no_ratings)


if __name__ == "__main__":
    main()
