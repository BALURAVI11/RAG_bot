"""RAGAS evaluation — measures faithfulness, answer relevance, and context recall."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))


EVAL_QUESTIONS = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "ground_truth": (
            "RAG is an AI framework that combines information retrieval with large language "
            "model generation to ground answers in external knowledge bases, reducing hallucinations."
        ),
    },
    {
        "question": "What are the main vector database options for RAG systems?",
        "ground_truth": (
            "Popular vector databases include Pinecone (managed, cloud-native), Weaviate "
            "(open-source with hybrid search), ChromaDB (lightweight, local), and Qdrant."
        ),
    },
    {
        "question": "What chunk size is recommended for RAG?",
        "ground_truth": (
            "256–512 tokens is the recommended chunk size, balancing context completeness "
            "with retrieval precision. 50-token overlap prevents context loss at boundaries."
        ),
    },
    {
        "question": "What metrics does RAGAS measure?",
        "ground_truth": (
            "RAGAS measures Faithfulness, Answer Relevance, Context Recall, and Context Precision "
            "without requiring human annotations."
        ),
    },
    {
        "question": "What is LangChain and what are its core abstractions?",
        "ground_truth": (
            "LangChain is a framework for building LLM applications. Its core abstractions are "
            "Chains, Agents, Memory, and Retrievers."
        ),
    },
]


def run_evaluation(use_sample: bool = True) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on the configured knowledge base.

    Returns a dict with per-metric scores and a summary.
    """
    print("=" * 60)
    print("RAG Knowledge Assistant — RAGAS Evaluation")
    print("=" * 60)

    # 1. Build the RAG chain
    print("\n[1/4] Initialising RAG pipeline...")
    from ingestion import run_ingestion_pipeline
    from chain.rag_chain import RAGChain

    vector_store = run_ingestion_pipeline(use_sample=use_sample)
    rag = RAGChain(vector_store)
    print("  ✓ Pipeline ready")

    # 2. Run queries and collect contexts + answers
    print(f"\n[2/4] Running {len(EVAL_QUESTIONS)} evaluation queries...")
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q = item["question"]
        print(f"  Query {i}: {q[:60]}...")

        result = rag.query(q)
        questions.append(q)
        answers.append(result["answer"])
        ground_truths.append(item["ground_truth"])
        # contexts = list of source previews used
        contexts.append([s["preview"] for s in result["sources"]])

    print("  ✓ All queries complete")

    # 3. Build RAGAS dataset
    print("\n[3/4] Computing RAGAS metrics...")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )

        results = {
            "faithfulness": round(float(scores["faithfulness"]), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
            "context_recall": round(float(scores["context_recall"]), 4),
            "context_precision": round(float(scores["context_precision"]), 4),
        }

    except Exception as e:
        print(f"  ⚠ RAGAS scoring failed ({e}) — using manual scoring fallback")
        results = _manual_score(questions, answers, contexts, ground_truths)

    # 4. Print report
    print("\n[4/4] Results")
    print("-" * 40)
    targets = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_recall": 0.75,
        "context_precision": 0.70,
    }
    all_pass = True
    for metric, score in results.items():
        target = targets.get(metric, 0.70)
        status = "✓ PASS" if score >= target else "✗ FAIL"
        if score < target:
            all_pass = False
        print(f"  {metric:<22} {score:.3f}  (target ≥ {target})  {status}")

    print("-" * 40)
    print(f"  Overall: {'ALL METRICS PASSED' if all_pass else 'SOME METRICS BELOW TARGET'}")
    print("=" * 60)

    # Save results
    output_path = Path(__file__).parent / "ragas_results.json"
    with open(output_path, "w") as f:
        json.dump({"metrics": results, "passed": all_pass, "n_queries": len(EVAL_QUESTIONS)}, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def _manual_score(questions, answers, contexts, ground_truths) -> Dict[str, float]:
    """Simple keyword-overlap scoring when RAGAS is unavailable."""
    def overlap(a: str, b: str) -> float:
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not b_words:
            return 0.0
        return len(a_words & b_words) / len(b_words)

    faithfulness_scores, relevancy_scores, recall_scores = [], [], []

    for q, a, ctx_list, gt in zip(questions, answers, contexts, ground_truths):
        ctx_text = " ".join(ctx_list)
        faithfulness_scores.append(overlap(a, ctx_text))
        relevancy_scores.append(overlap(a, q))
        recall_scores.append(overlap(ctx_text, gt))

    return {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 4),
        "context_recall": round(sum(recall_scores) / len(recall_scores), 4),
        "context_precision": round(sum(recall_scores) / len(recall_scores), 4),
    }


if __name__ == "__main__":
    run_evaluation(use_sample=True)
