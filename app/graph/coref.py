"""Coreference resolution for graph extraction — resolves pronouns and aliases
to canonical entity names before sending chunks to GPT-4o.

Uses fastcoref (FCoref model). Degrades gracefully if unavailable.
"""

import asyncio
import logging
from typing import Optional

from app.ingestion.chunker import Chunk, _token_count

logger = logging.getLogger(__name__)

_model = None
_coref_available: Optional[bool] = None  # None=unchecked, True/False


def _get_model():
    """Lazy-load fastcoref FCoref model. Returns None if unavailable."""
    global _model, _coref_available

    if _coref_available is False:
        return None
    if _model is not None:
        return _model

    try:
        from fastcoref import FCoref
        _model = FCoref(device="cpu")
        _coref_available = True
        logger.info("fastcoref loaded successfully")
        return _model
    except Exception as exc:
        _coref_available = False
        logger.warning("Coreferee unavailable, coreference resolution disabled: %s", exc)
        return None


def _best_mention(cluster: list[tuple[int, int]], text: str) -> str:
    """Pick the longest mention from a cluster as the canonical form."""
    mentions = [(text[start:end], end - start) for start, end in cluster]
    return max(mentions, key=lambda m: m[1])[0]


def _apply_replacements(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply character-offset replacements right-to-left to preserve offsets."""
    for start, end, replacement in sorted(replacements, key=lambda r: r[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _resolve_chunk_window(prev_content: Optional[str], current_content: str, model) -> str:
    """Run fastcoref on [prev_content + current_content], replace only within current span."""
    separator = "\n\n"
    if prev_content:
        combined = prev_content + separator + current_content
        current_start = len(prev_content) + len(separator)
    else:
        combined = current_content
        current_start = 0

    preds = model.predict(texts=[combined])
    clusters = preds[0].get_clusters(as_strings=False)

    if not clusters:
        return current_content

    replacements: list[tuple[int, int, str]] = []

    for cluster in clusters:
        best = _best_mention(cluster, combined)

        for start, end in cluster:
            # Only replace within current_content's char span
            if start < current_start:
                continue

            mention_text = combined[start:end]

            # Only replace short mentions (pronouns / short aliases)
            if len(mention_text.split()) > 3:
                continue

            if mention_text == best:
                continue

            # Adjust offsets relative to current_content
            rel_start = start - current_start
            rel_end = end - current_start
            replacements.append((rel_start, rel_end, best))

    if not replacements:
        return current_content

    return _apply_replacements(current_content, replacements)


async def resolve_coreferences(chunks: list[Chunk]) -> list[str]:
    """Resolve coreferences across chunks. Never raises.

    Returns resolved text per chunk (same length as input).
    Runs model inference in thread pool to avoid blocking the event loop.
    If fastcoref is unavailable, returns original content.
    """
    if not chunks:
        return []

    model = _get_model()
    if model is None:
        return [c.content for c in chunks]

    loop = asyncio.get_event_loop()
    results: list[str] = []

    for i, chunk in enumerate(chunks):
        if _token_count(chunk.content) < 10:
            results.append(chunk.content)
            continue

        prev_content = chunks[i - 1].content if i > 0 else None

        try:
            resolved = await loop.run_in_executor(
                None, _resolve_chunk_window, prev_content, chunk.content, model,
            )
            results.append(resolved)
        except Exception as exc:
            logger.warning("Coref failed for chunk %d, using original: %s", i, exc)
            results.append(chunk.content)

    return results
