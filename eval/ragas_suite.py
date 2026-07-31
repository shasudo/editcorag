"""Optional RAGAS scoring, run OUT-OF-BAND.

RAGAS is deliberately not a project dependency: it pulls datasets, pandas and
pyarrow for an optional metrics pass, and its own pins have repeatedly conflicted
with this project's. Instead `editcorag eval --export runs/latest.jsonl`
writes the record shape and this script scores it in its own throwaway
environment:

    editcorag eval --export runs/latest.jsonl
    uvx --with ragas --with datasets --with langchain-openai \\
        python eval/ragas_suite.py runs/latest.jsonl

RAGAS is pointed at the same Groq model the app answers with, so the judge and
the system under test agree on capability. GROQ_API_KEY must be set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    """Score an exported eval run with RAGAS, excluding abstentions.

    Faithfulness and answer relevancy are undefined without retrieved contexts,
    and an abstention scored anyway lands at zero - a refusal that behaved
    correctly would read as a regression. A non-zero exit means the run could
    not be scored at all, never a low score.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"no such file: {path}")
        return 1

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    usable = [r for r in records if r.get("contexts") and not r.get("abstained")]
    print(f"{len(records)} record(s), {len(usable)} scoreable (abstentions excluded)")
    if not usable:
        return 0

    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        print("dependencies missing; run this via uvx as shown in the module docstring")
        return 2

    model = os.environ.get("LOCALRAG_LLM_MODEL", "openai/gpt-oss-120b").split(":", 1)[-1]
    # The same local embedder the index was built with, so retrieval-derived
    # metrics are scored in the space the passages actually live in.
    embed_id = os.environ.get("LOCALRAG_EMBED_MODEL", "BAAI/bge-m3")
    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
            }
            for r in usable
        ]
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ChatOpenAI(
            model=model,
            base_url=os.environ.get("LOCALRAG_LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.environ.get("GROQ_API_KEY") or os.environ["LOCALRAG_LLM_API_KEY"],
            temperature=0.0,
        ),
        embeddings=HuggingFaceEmbeddings(model_name=embed_id),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
