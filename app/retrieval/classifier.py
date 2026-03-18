"""Query classification — GPT-5.4 classifies query into QueryType + expansion."""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from app.config import settings
from app.retrieval.models import ExecutionPlan, QueryType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a query classifier for a RAG system over Indian legal statutes. Given a user query, classify it and expand it.

Return JSON with exactly these keys:
- "query_type": one of "SIMPLE", "FILTERED", "GRAPH", "ANALYTICAL"
- "expanded_queries": array of 1-3 query reformulations for better retrieval
- "metadata_filters": object with optional keys: "document_id" (UUID only), "filename", or "section_path" (chapter/section name) if query references a specific document or section

Classification rules:
- SIMPLE: single fact lookup, definition, or yes/no question (e.g. "What is X?", "Define Y")
- FILTERED: the user wants to READ a specific passage or section — not analyze it. The query only needs information from that narrow scope.
- GRAPH: relationship-based, entity connections, multi-entity queries (e.g. "How does X relate to Y?", "Who is responsible for X under Y?"). Includes queries about obligations between parties, regulatory relationships, entity roles.
- ANALYTICAL: multi-hop reasoning, comparison, aggregation, contradiction analysis, conditional reasoning, exhaustive listing, counterfactual reasoning. This is the broadest activation — when in doubt between FILTERED and ANALYTICAL, prefer ANALYTICAL.

IMPORTANT disambiguation:
- Section or provision references do NOT automatically mean FILTERED. "What are all the remedies under Section 18?" is ANALYTICAL (aggregation). "What does Section 3(1) say?" is FILTERED (reading).
- If the query asks about contradiction, conflict, or proviso vs. main text, classify as ANALYTICAL.
- If the query requires traversing relationships between legal entities (buyer, promoter, authority, tribunal), classify as GRAPH.
- Queries asking "all the...", "what conditions", "what would be different if", or exhaustive lists are ANALYTICAL.

Examples:
- "What is the definition of 'promoter' under RERA?" → SIMPLE
- "What does Section 3 say?" → FILTERED
- "What are all the legal remedies available to a buyer who paid advance?" → ANALYTICAL
- "How does the promoter's obligation relate to the buyer's rights?" → GRAPH
- "Section 11(4)(a) states X. The proviso says Y. How do these interact?" → ANALYTICAL
- "Compare the penalties in Chapter VII with the remedies in Chapter VIII" → ANALYTICAL
- "Under what conditions can the authority revoke a registration?" → ANALYTICAL
- "What are the promoter's obligations regarding insurance?" → ANALYTICAL
- "What is the timeline for possession under Section 18?" → FILTERED

Common mistakes to AVOID:
- "What are all the remedies..." is NOT FILTERED — it requires aggregating across multiple provisions → ANALYTICAL
- "Section X says... but the proviso says..." is NOT FILTERED — it requires contradiction analysis → ANALYTICAL
- "What happens if the buyer suffers financial loss?" is NOT FILTERED — it requires multi-hop reasoning → GRAPH or ANALYTICAL

Expanded query generation rules:
- Always include the original query as-is
- For queries about rights/remedies of one party, add a query about the OBLIGATIONS of the counter-party
- For queries about penalties, add a query about the VIOLATIONS that trigger them
- Use legal synonym expansion: "buyer" → "allottee/purchaser", "advance" → "deposit/amount paid"

Return ONLY valid JSON, no markdown fences."""


# ---------------------------------------------------------------------------
# Heuristic pre-check (saves LLM cost on obvious cases)
# ---------------------------------------------------------------------------

_CALC_PATTERN = re.compile(
    r"\b(calculate|sum|average|total|mean|percentage|ratio|difference)\b.*\d",
    re.IGNORECASE,
)
_ANALYTICAL_PATTERN = re.compile(
    r"\b("
    r"all\s+(?:the\s+)?(?:remedies|provisions|conditions|obligations|rights|penalties|exceptions)"
    r"|what\s+(?:would|could|should)\s+(?:be|happen)\s+(?:different|if)"
    r"|under\s+what\s+conditions"
    r"|compare|contrast|distinguish"
    r"|proviso\s+(?:says|states|provides)"
    r"|how\s+do\s+(?:these|they|the\s+\w+)\s+interact"
    r"|exhaustive|comprehensive\s+list"
    r"|contradiction|conflict\s+between"
    r"|what\s+are\s+(?:the\s+)?(?:promoter'?s?|buyer'?s?|authority'?s?)\s+obligations"
    r"|obligations?\s+(?:regarding|concerning|related\s+to|under)"
    r")\b",
    re.IGNORECASE,
)
_GRAPH_PATTERN = re.compile(
    r"\b(relationship|connected|relate[sd]?\s+to|link(?:ed|s)?\s+(?:to|with)|how does .+ (?:affect|impact|influence))\b",
    re.IGNORECASE,
)
_CONFLICT_PATTERN = re.compile(
    r"\b("
    r"contradict[s]?|conflicting?"
    r"|proviso\s+says|but\s+the\s+same\s+section"
    r"|reconcil[e]|apparent\s+tension"
    r"|inconsisten[ct]"
    r"|notwithstanding"
    r"|contrary\s+to"
    r")\b",
    re.IGNORECASE,
)


def _heuristic_classify(query: str) -> QueryType | None:
    """Return QueryType if obvious from heuristics, else None (call LLM)."""
    if _CALC_PATTERN.search(query):
        return QueryType.ANALYTICAL
    if _ANALYTICAL_PATTERN.search(query):
        return QueryType.ANALYTICAL
    if _GRAPH_PATTERN.search(query):
        return QueryType.GRAPH
    if _CONFLICT_PATTERN.search(query):
        return QueryType.GRAPH
    return None


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

async def classify_query(query: str) -> ExecutionPlan:
    """Classify query and build ExecutionPlan. Never raises — defaults to SIMPLE."""
    # Try heuristic first
    heuristic = _heuristic_classify(query)

    query_type = QueryType.SIMPLE
    expanded: list[str] = [query]
    filters: dict = {}

    # Always call LLM for expanded queries; heuristic overrides query_type only
    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=settings.QUERY_CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=512,
        )
        data = json.loads(resp.choices[0].message.content)

        expanded = data.get("expanded_queries", [query]) or [query]
        filters = data.get("metadata_filters", {}) or {}

        if heuristic:
            query_type = heuristic
        else:
            qt = data.get("query_type", "SIMPLE").upper()
            if qt in QueryType.__members__:
                query_type = QueryType(qt)

    except Exception as exc:
        logger.warning("Query classification failed, defaulting: %s", exc)
        if heuristic:
            query_type = heuristic

    # Build plan
    plan = ExecutionPlan(
        query_type=query_type,
        activate_vector=True,
        activate_graph=query_type in (QueryType.GRAPH, QueryType.ANALYTICAL),
        activate_calculator=bool(_CALC_PATTERN.search(query)),
        activate_conflict=True,
        metadata_filters=filters,
        expanded_queries=expanded,
    )

    # Safety net: force graph activation on explicit conflict language
    if not plan.activate_graph and _CONFLICT_PATTERN.search(query):
        plan.activate_graph = True
        logger.info("Conflict language detected — forcing graph activation")

    logger.info("Classified query as %s (graph=%s, calc=%s)", query_type.value, plan.activate_graph, plan.activate_calculator)
    return plan
