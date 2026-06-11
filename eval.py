"""
eval.py

Evaluates JobLens retrieval (and optionally generation) against
eval_questions.json — 20 questions with ground truth computed
directly from the Job Tracker.

What it measures:
  1. Retrieval hit-rate: do the top-k retrieved chunks contain the
     expected keywords? (No LLM needed — fast, deterministic.)
  2. (--llm flag) Full answers from Llama via Ollama, printed for
     manual faithfulness review against the 'truth' field.

Run:    python eval.py            # retrieval only
        python eval.py --llm      # retrieval + generated answers
"""

import json
import re
import sys

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "joblens_index"
QUESTIONS_PATH = "eval_questions.json"
TOP_K = 5


def keyword_hit(kw, text):
    """Word-boundary match so 'rag' can't match inside 'storage'."""
    pat = r"(?<![a-z0-9])" + re.escape(kw.lower()) + r"(?![a-z0-9])"
    return re.search(pat, text.lower()) is not None


def main():
    use_llm = "--llm" in sys.argv

    with open(QUESTIONS_PATH) as f:
        spec = json.load(f)
    questions = spec["questions"]

    print("Loading embedding model + FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vs = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    chain = None
    if use_llm:
        from rag import load_qa_chain, ask
        chain = load_qa_chain()

    results = []
    for q in questions:
        docs = retriever.invoke(q["question"])
        blob = " ".join(d.page_content for d in docs)
        hits = [kw for kw in q["expect_keywords"] if keyword_hit(kw, blob)]
        passed = (
            len(hits) == len(q["expect_keywords"])
            if q["match"] == "all"
            else len(hits) > 0
        )
        results.append((q, passed, hits))

        mark = "PASS" if passed else "MISS"
        print(f"[{mark}] Q{q['id']:>2} ({q['type']:9}) {q['question']}")
        if not passed:
            missing = [k for k in q["expect_keywords"] if k not in hits]
            print(f"        missing from top-{TOP_K} chunks: {missing}")
        if use_llm and chain:
            answer, _ = ask(chain, q["question"])
            print(f"        LLM answer: {answer.strip()[:300]}")
            print(f"        truth:      {q['truth']}")

    # ---- summary ----
    total = len(results)
    passed_n = sum(1 for _, p, _ in results if p)
    print("\n" + "=" * 60)
    print(f"Retrieval hit-rate: {passed_n}/{total} ({passed_n/total:.0%})")
    for t in ("lookup", "existence", "aggregate"):
        sub = [(q, p) for q, p, _ in results if q["type"] == t]
        if sub:
            ok = sum(1 for _, p in sub if p)
            print(f"  {t:9}: {ok}/{len(sub)}")
    print("=" * 60)
    print(
        "\nNote: aggregate questions are EXPECTED to be weaker — top-k\n"
        "retrieval can't count across all 172 JDs. For Q18 and Q20 the\n"
        "correct LLM behavior is 'I don't have enough data.' Check that\n"
        "with --llm. That honesty is a feature, not a bug."
    )


if __name__ == "__main__":
    main()
