"""Contradiction detection + credibility-based resolution."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.retrieval.models import ConflictResolution, RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Analyze the provided text chunks for contradictions.

Two claims contradict ONLY if they assert incompatible facts about the SAME entity performing the SAME action or having the SAME attribute.

NOT contradictions (do NOT flag these):
- Different entities having different penalties, obligations, or rights (e.g., promoter penalty ≠ agent penalty)
- Different sections describing different requirements
- General rule vs. specific exception (these complement each other)
- Different time periods or conditions with different outcomes

Return JSON with exactly this key:
- "conflicts": array of objects, each with:
  - "chunk_a_id": chunk ID of first claim
  - "chunk_b_id": chunk ID of second claim
  - "claim_a": the first claim (short quote)
  - "claim_b": the contradicting claim (short quote)

If there are no contradictions, return {"conflicts": []}.
Return ONLY valid JSON, no markdown fences."""


def _resolve_winner(
    chunk_a: RetrievedChunk,
    chunk_b: RetrievedChunk,
) -> tuple[RetrievedChunk, str]:
    """Determine which chunk wins based on credibility hierarchy."""
    # Higher version wins
    if chunk_a.version != chunk_b.version:
        winner = chunk_a if chunk_a.version > chunk_b.version else chunk_b
        return winner, f"version {winner.version} > version {min(chunk_a.version, chunk_b.version)}"

    # Newer ingestion wins
    if chunk_a.ingested_at and chunk_b.ingested_at and chunk_a.ingested_at != chunk_b.ingested_at:
        winner = chunk_a if chunk_a.ingested_at > chunk_b.ingested_at else chunk_b
        return winner, "newer ingestion date"

    # Table content beats paragraph (tables are more precise)
    a_table = chunk_a.metadata.get("has_table", False)
    b_table = chunk_b.metadata.get("has_table", False)
    if a_table != b_table:
        winner = chunk_a if a_table else chunk_b
        return winner, "table content is more precise than paragraph"

    # Higher reranking score
    winner = chunk_a if chunk_a.score >= chunk_b.score else chunk_b
    return winner, "higher relevance score"


async def detect_and_resolve_conflicts(
    chunks: list[RetrievedChunk],
) -> list[ConflictResolution]:
    """Detect contradictions and resolve via credibility hierarchy. Never raises."""
    if len(chunks) < 2:
        return []

    try:
        context = "\n\n".join(
            f"[{c.chunk_id[:12]}] (file: {c.filename}, v{c.version}):\n{c.content[:600]}"
            for c in chunks[:10]
        )

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=settings.CONFLICT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Chunks:\n{context}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=1024,
        )
        data = json.loads(resp.choices[0].message.content)

        conflicts_raw = data.get("conflicts", [])
        if not conflicts_raw:
            return []

        # Build chunk lookup (by truncated ID prefix)
        chunk_map: dict[str, RetrievedChunk] = {}
        for c in chunks:
            chunk_map[c.chunk_id[:12]] = c
            chunk_map[c.chunk_id] = c

        resolutions: list[ConflictResolution] = []
        for conflict in conflicts_raw:
            a_id = conflict.get("chunk_a_id", "")
            b_id = conflict.get("chunk_b_id", "")
            chunk_a = chunk_map.get(a_id)
            chunk_b = chunk_map.get(b_id)

            if not chunk_a or not chunk_b:
                continue

            winner, reason = _resolve_winner(chunk_a, chunk_b)
            resolutions.append(ConflictResolution(
                claim_a=conflict.get("claim_a", ""),
                claim_b=conflict.get("claim_b", ""),
                resolution=f"Trusting chunk {winner.chunk_id[:12]}",
                winner_chunk_id=winner.chunk_id,
                reason=reason,
            ))

        logger.info("Detected %d conflicts, resolved %d", len(conflicts_raw), len(resolutions))
        return resolutions

    except Exception as exc:
        logger.warning("Conflict detection failed: %s", exc)
        return []
