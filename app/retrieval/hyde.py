"""Hypothetical Document Embeddings (HyDE) — generate a hypothetical answer
passage for non-SIMPLE queries to bridge the vocabulary gap between user
queries and legal document language."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import settings
from app.retrieval.models import QueryType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a legal document passage generator. Given a user question about a legal statute or regulation, write a SHORT passage (3-5 sentences) that might appear in the actual legal document and would answer this question.

Rules:
- Write in formal legal document style, using statutory language
- Use terms that would appear in the actual legislation (e.g. "person" not "buyer", "deposit" not "advance payment", "sustains loss or damage" not "suffers financial loss")
- Include section-style references if the question mentions them
- Do NOT answer the question — write what the DOCUMENT would say
- Keep it concise: 3-5 sentences maximum"""


async def generate_hyde_passage(
    client: AsyncOpenAI,
    query: str,
    query_type: QueryType,
) -> str | None:
    """Generate a hypothetical document passage for the query.

    Returns None for SIMPLE queries or on failure. Never raises.
    """
    if query_type == QueryType.SIMPLE:
        return None

    if not settings.HYDE_ENABLED:
        return None

    try:
        resp = await client.chat.completions.create(
            model=settings.HYDE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_completion_tokens=settings.HYDE_MAX_TOKENS,
        )
        passage = resp.choices[0].message.content.strip()
        logger.info("HyDE passage generated (%d chars) for query type %s", len(passage), query_type.value)
        return passage

    except Exception as exc:
        logger.warning("HyDE generation failed, falling back to raw query: %s", exc)
        return None
