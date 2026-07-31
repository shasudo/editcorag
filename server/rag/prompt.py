"""Prompts and structured-output schemas for the query graph.

Grader schemas are deliberately tiny. A local 8B model holds a two-field object
reliably and starts drifting on anything elaborate, and every field costs a
round trip's worth of latency on a laptop.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ANSWER_SYSTEM = """\
You answer questions strictly from the provided excerpts of a single document.

Rules:
- Use ONLY the excerpts. Never add outside knowledge, and never guess.
- After every sentence containing a fact, cite the excerpt it came from using
  its bracketed id, e.g. [S1]. Cite several like [S1][S3] when relevant.
- If the excerpts do not contain the answer, reply exactly:
  NOT_IN_DOCUMENT
  and nothing else.
- Quote exact figures, units and identifiers as written. Do not round or convert.
- Be concise and factual. No preamble, no restating the question.
"""

ANSWER_USER = """\
Question: {question}

Excerpts:
{context}

Answer with citations:"""

STRICTER_ANSWER_USER = """\
Question: {question}

Excerpts:
{context}

Your previous answer contained statements not supported by the excerpts.
Write a new answer that asserts ONLY what the excerpts state verbatim. Cite an
excerpt id after every factual sentence. If support is missing, reply exactly
NOT_IN_DOCUMENT.

Answer with citations:"""


CONDENSE_SYSTEM = """\
Rewrite the user's latest message into a standalone search query.

- Resolve pronouns and references using the conversation ("what about the second
  one?" -> the thing it refers to).
- Keep every identifier, number, unit and proper noun exactly as written; they
  are what the search matches on.
- Output only the rewritten query, no explanation.
- If the message is already standalone, output it unchanged.
"""

CONDENSE_USER = """\
Conversation:
{history}

Latest message: {question}

Standalone query:"""


REWRITE_SYSTEM = """\
A document search returned nothing useful for this query. Rewrite it to search
better over a technical document.

- Try the terminology a document would use rather than the terminology a person
  used; expand abbreviations, and add an obvious synonym or two.
- Keep all identifiers and numbers.
- Output only the rewritten query.
"""

REWRITE_USER = """\
Original query: {question}
{tried}
Rewritten query:"""


class DocumentRelevance(BaseModel):
    """Whether the retrieved excerpts can answer the question at all."""

    relevant: bool = Field(
        description="true if at least one excerpt contains information that answers the question"
    )
    reason: str = Field(description="one short sentence explaining the decision")


GRADE_DOCS_SYSTEM = """\
Decide whether the excerpts contain information that answers the question.

Judge only whether the needed information is present. Do not judge writing
quality, completeness of the document, or whether you personally know the answer.
Related-but-not-answering excerpts count as NOT relevant.
"""

GRADE_DOCS_USER = """\
Question: {question}

Excerpts:
{context}"""


class Groundedness(BaseModel):
    """Whether a drafted answer is fully supported by the excerpts."""

    grounded: bool = Field(
        description="true if every factual claim in the answer is stated in the excerpts"
    )
    unsupported: str = Field(
        default="", description="the first unsupported claim, or empty if none"
    )


GROUNDED_SYSTEM = """\
Check whether the answer is fully supported by the excerpts.

Mark it ungrounded if it states any fact, number, unit or name that the
excerpts do not contain, even if that fact is true in general. Rephrasing and
summarising the excerpts is fine and counts as grounded.
"""

GROUNDED_USER = """\
Excerpts:
{context}

Answer:
{answer}"""


NOT_IN_DOCUMENT = "NOT_IN_DOCUMENT"

ABSTAIN_MESSAGE = (
    "I could not find this in the indexed document. "
    "The closest passages are listed below, but none of them answer the question."
)


def format_history(history: list[dict[str, str]], limit: int = 6) -> str:
    """Render recent turns for the condense prompt."""
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history[-limit:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(no previous turns)"
