"""Compress retrieved chunks into a structured answer. GPT-5.4."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import settings
from app.retrieval.models import GraphPath, RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context chunks.

Rules:
- Use ONLY information from the provided chunks — do not add outside knowledge
- Cite chunk IDs in brackets like [chunk_abc123] when referencing specific information
- If the context is insufficient to fully answer, explicitly state what information is missing
- Be concise but thorough — include all relevant details from the context
- If chunks contain conflicting information, note the conflict and which source you trust more
- Structure your answer with clear paragraphs for complex responses
- Pay special attention to Explanation clauses and Provisos — these define or qualify key terms and MUST be included in your answer when relevant"""


_PRIORITY_CHUNK_TYPES = {"DEFINITION"}


def _build_context(
    chunks: list[RetrievedChunk],
    graph_paths: list[GraphPath],
    max_tokens: int,
) -> str:
    """Build context string with 2-tier priority: definitions → other by score."""
    definition_chunks = [
        c for c in chunks
        if c.metadata.get("chunk_type") in _PRIORITY_CHUNK_TYPES
    ]
    other_chunks = [
        c for c in chunks
        if c.metadata.get("chunk_type") not in _PRIORITY_CHUNK_TYPES
    ]

    definition_chunks.sort(key=lambda c: c.score, reverse=True)
    other_chunks.sort(key=lambda c: c.score, reverse=True)

    ordered = definition_chunks + other_chunks
    parts: list[str] = []
    budget = max_tokens * 4  # rough char estimate

    for chunk in ordered:
        entry = (
            f"[{chunk.chunk_id[:12]}] "
            f"(file: {chunk.filename}, section: {chunk.section_path})\n"
            f"{chunk.content}"
        )
        if len("\n\n".join(parts + [entry])) > budget:
            break
        parts.append(entry)

    return "\n\n".join(parts)


async def summarise_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    graph_paths: list[GraphPath] | None = None,
    calculation_result: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Compress chunks into final answer. Never raises — returns error message on failure."""
    max_tokens = max_tokens or settings.SUMMARISER_MAX_TOKENS
    graph_paths = graph_paths or []

    if not chunks:
        return "I could not find any relevant information to answer this question."

    try:
        context = _build_context(chunks, graph_paths, max_tokens)

        user_msg = f"Question: {query}\n\nContext:\n{context}"
        if calculation_result:
            user_msg += f"\n\nCalculation result: {calculation_result}"

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=settings.SUMMARISER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=max_tokens,
        )

        answer = resp.choices[0].message.content.strip()
        logger.info("Summarised %d chunks into %d-char answer", len(chunks), len(answer))
        return answer

    except Exception as exc:
        logger.warning("Summarisation failed: %s", exc)
        return f"I retrieved {len(chunks)} relevant chunks but failed to generate a summary: {exc}"
