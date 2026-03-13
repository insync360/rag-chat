"""Safe arithmetic — structured extraction + Python math. No exec/eval."""

from __future__ import annotations

import json
import logging
import math
import statistics

from openai import AsyncOpenAI

from app.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Extract numbers and the requested operation from the query and context.

Return JSON with exactly these keys:
- "numbers": array of floats extracted from the context
- "operation": one of "sum", "mean", "difference", "ratio", "percentage_change", "min", "max", "count"
- "context": brief description of what the numbers represent
- "applicable": boolean — false if the query doesn't require calculation

Return ONLY valid JSON, no markdown fences."""

_OPERATIONS: dict[str, callable] = {
    "sum": sum,
    "mean": statistics.mean,
    "difference": lambda nums: nums[0] - nums[1] if len(nums) >= 2 else 0.0,
    "ratio": lambda nums: nums[0] / nums[1] if len(nums) >= 2 and nums[1] != 0 else 0.0,
    "percentage_change": lambda nums: ((nums[1] - nums[0]) / nums[0]) * 100 if len(nums) >= 2 and nums[0] != 0 else 0.0,
    "min": min,
    "max": max,
    "count": lambda nums: float(len(nums)),
}


async def calculate(
    query: str,
    chunks: list[RetrievedChunk],
) -> str | None:
    """Extract numbers + operation from query/chunks, compute result. Never raises."""
    if not chunks:
        return None

    try:
        context = "\n".join(
            f"[{c.chunk_id[:12]}]: {c.content[:500]}" for c in chunks[:10]
        )

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=settings.CALCULATOR_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=512,
        )
        data = json.loads(resp.choices[0].message.content)

        if not data.get("applicable", True):
            return None

        numbers = [float(n) for n in data.get("numbers", [])]
        operation = data.get("operation", "sum")
        desc = data.get("context", "")

        if not numbers:
            return None

        op_fn = _OPERATIONS.get(operation)
        if op_fn is None:
            return None

        result = op_fn(numbers)
        # Round to reasonable precision
        if isinstance(result, float):
            result = round(result, 4)

        answer = f"Calculation ({operation}): {result}"
        if desc:
            answer += f" — {desc}"
        answer += f" (from numbers: {numbers})"

        logger.info("Calculator: %s(%s) = %s", operation, numbers, result)
        return answer

    except Exception as exc:
        logger.warning("Calculator failed: %s", exc)
        return None
