"""Coreference resolution for graph extraction — resolves pronouns and aliases
to canonical entity names before sending chunks to GPT-4o.

Uses coreferee (spaCy 3.x pipeline component). Degrades gracefully if unavailable.
"""

import asyncio
import logging
from typing import Optional

from app.ingestion.chunker import Chunk, _token_count

logger = logging.getLogger(__name__)

_nlp = None
_coref_available: Optional[bool] = None  # None=unchecked, True/False


def _get_nlp():
    """Lazy-load spaCy + coreferee. Returns None if unavailable."""
    global _nlp, _coref_available

    if _coref_available is False:
        return None
    if _nlp is not None:
        return _nlp

    try:
        import spacy
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("coreferee")
        _nlp = nlp
        _coref_available = True
        logger.info("Coreferee loaded successfully")
        return _nlp
    except Exception as exc:
        _coref_available = False
        logger.warning("Coreferee unavailable, coreference resolution disabled: %s", exc)
        return None


def _best_mention(chain, doc) -> str:
    """Pick the most informative mention from a coreference chain.

    Priority: proper nouns > longest non-pronoun > first mention.
    """
    mentions = []
    for mention in chain:
        indices = mention.token_indexes if hasattr(mention, "token_indexes") else [mention.root_index]
        span_text = " ".join(doc[i].text for i in indices)
        has_proper = any(doc[i].pos_ == "PROPN" for i in indices)
        is_pronoun = all(doc[i].pos_ == "PRON" for i in indices)
        mentions.append((span_text, has_proper, is_pronoun, len(indices)))

    # Prefer proper nouns, then longest non-pronoun, then first
    proper = [m for m in mentions if m[1]]
    if proper:
        return max(proper, key=lambda m: m[3])[0]
    non_pron = [m for m in mentions if not m[2]]
    if non_pron:
        return max(non_pron, key=lambda m: m[3])[0]
    return mentions[0][0]


def _apply_replacements(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply character-offset replacements right-to-left to preserve offsets."""
    for start, end, replacement in sorted(replacements, key=lambda r: r[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _resolve_chunk_window(prev_content: Optional[str], current_content: str, nlp) -> str:
    """Run coreferee on [prev_content + current_content], replace only within current span."""
    separator = "\n\n"
    if prev_content:
        combined = prev_content + separator + current_content
        current_start = len(prev_content) + len(separator)
    else:
        combined = current_content
        current_start = 0

    doc = nlp(combined)

    if not doc._.coref_chains:
        return current_content

    replacements: list[tuple[int, int, str]] = []

    for chain in doc._.coref_chains:
        best = _best_mention(chain, doc)

        for mention in chain:
            indices = mention.token_indexes if hasattr(mention, "token_indexes") else [mention.root_index]
            span_start = doc[indices[0]].idx
            last_tok = doc[indices[-1]]
            span_end = last_tok.idx + len(last_tok.text)

            # Only replace within current_content's char span
            if span_start < current_start:
                continue

            # Only replace pronouns and short aliases (<=3 tokens)
            is_pronoun = all(doc[i].pos_ == "PRON" for i in indices)
            if not is_pronoun and len(indices) > 3:
                continue

            mention_text = " ".join(doc[i].text for i in indices)
            if mention_text == best:
                continue

            # Adjust offsets relative to current_content
            rel_start = span_start - current_start
            rel_end = span_end - current_start
            replacements.append((rel_start, rel_end, best))

    if not replacements:
        return current_content

    return _apply_replacements(current_content, replacements)


async def resolve_coreferences(chunks: list[Chunk]) -> list[str]:
    """Resolve coreferences across chunks. Never raises.

    Returns resolved text per chunk (same length as input).
    Runs spaCy in thread pool to avoid blocking the event loop.
    If coreferee is unavailable, returns original content.
    """
    if not chunks:
        return []

    nlp = _get_nlp()
    if nlp is None:
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
                None, _resolve_chunk_window, prev_content, chunk.content, nlp,
            )
            results.append(resolved)
        except Exception as exc:
            logger.warning("Coref failed for chunk %d, using original: %s", i, exc)
            results.append(chunk.content)

    return results
