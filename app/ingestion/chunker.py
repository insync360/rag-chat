"""Structure-aware chunker — splits markdown into semantically coherent chunks.

Respects heading boundaries, table/code block integrity, and paragraph coherence.
Three phases: parse blocks → group into chunks → apply overlap.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto

import tiktoken

from app.config import settings
from app.database import get_pool

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class BlockType(Enum):
    HEADING = auto()
    TABLE = auto()
    CODE = auto()
    PARAGRAPH = auto()


@dataclass
class Block:
    type: BlockType
    content: str
    heading_level: int = 0   # 1-6 for HEADING, 0 otherwise
    heading_text: str = ""   # raw heading text without #


@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    section_path: str       # "HR Policy > California > Parental Leave"
    has_table: bool
    has_code: bool
    overlap_tokens: int
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""


# ---------------------------------------------------------------------------
# Phase 1: Parse markdown into typed blocks
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^```")
_TABLE_ROW_RE = re.compile(r"^\|.*\|")


def _parse_blocks(markdown: str) -> list[Block]:
    lines = markdown.split("\n")
    blocks: list[Block] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            blocks.append(Block(
                type=BlockType.HEADING,
                content=line,
                heading_level=len(m.group(1)),
                heading_text=m.group(2).strip(),
            ))
            i += 1
            continue

        # Code fence
        if _CODE_FENCE_RE.match(line):
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if _CODE_FENCE_RE.match(lines[i]) and len(code_lines) > 1:
                    i += 1
                    break
                i += 1
            blocks.append(Block(type=BlockType.CODE, content="\n".join(code_lines)))
            continue

        # Table rows
        if _TABLE_ROW_RE.match(line):
            table_lines = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(Block(type=BlockType.TABLE, content="\n".join(table_lines)))
            continue

        # Paragraph (accumulate contiguous non-empty lines)
        if line.strip():
            para_lines = []
            while i < len(lines) and lines[i].strip() and not _HEADING_RE.match(lines[i]) and not _CODE_FENCE_RE.match(lines[i]) and not _TABLE_ROW_RE.match(lines[i]):
                para_lines.append(lines[i])
                i += 1
            blocks.append(Block(type=BlockType.PARAGRAPH, content="\n".join(para_lines)))
            continue

        # Blank line — skip
        i += 1

    return blocks


# ---------------------------------------------------------------------------
# Phase 2: Group blocks into raw chunks
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_paragraph(text: str, max_tokens: int) -> list[str]:
    """Split a long paragraph at sentence boundaries into pieces <= max_tokens."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_count(sent)
        if current and current_tokens + sent_tokens > max_tokens:
            pieces.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        pieces.append(" ".join(current))
    return pieces


@dataclass
class _RawChunk:
    parts: list[str] = field(default_factory=list)
    tokens: int = 0
    has_table: bool = False
    has_code: bool = False
    has_body: bool = False  # True once non-heading content is added


def _group_blocks(blocks: list[Block], document_id: str) -> list[Chunk]:
    min_tok = settings.CHUNK_MIN_TOKENS
    max_tok = settings.CHUNK_MAX_TOKENS

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current = _RawChunk()

    def section_path() -> str:
        return " > ".join(h[1] for h in heading_stack)

    def flush():
        nonlocal current
        if not current.parts:
            return
        content = "\n\n".join(current.parts)
        tokens = _token_count(content)
        chunks.append(Chunk(
            document_id=document_id,
            chunk_index=len(chunks),
            content=content,
            token_count=tokens,
            section_path=section_path(),
            has_table=current.has_table,
            has_code=current.has_code,
            overlap_tokens=0,
            metadata={},
        ))
        current = _RawChunk()

    for block in blocks:
        if block.type == BlockType.HEADING:
            # Only flush if current chunk has body content (not just headings)
            if current.has_body:
                flush()
            # Update heading stack
            while heading_stack and heading_stack[-1][0] >= block.heading_level:
                heading_stack.pop()
            heading_stack.append((block.heading_level, block.heading_text))
            # Heading text becomes start of new chunk
            current.parts.append(block.content)
            current.tokens += _token_count(block.content)
            continue

        block_tokens = _token_count(block.content)

        # Table/code: never split
        if block.type in (BlockType.TABLE, BlockType.CODE):
            # If adding would exceed max and current has content, flush first
            if current.parts and current.tokens + block_tokens > max_tok:
                flush()
            # Oversized table/code → standalone chunk
            if block_tokens > max_tok and not current.parts:
                current.parts.append(block.content)
                current.tokens = block_tokens
                current.has_body = True
                if block.type == BlockType.TABLE:
                    current.has_table = True
                else:
                    current.has_code = True
                flush()
                continue
            current.parts.append(block.content)
            current.tokens += block_tokens
            current.has_body = True
            if block.type == BlockType.TABLE:
                current.has_table = True
            else:
                current.has_code = True
            continue

        # Paragraph
        if block_tokens > max_tok:
            # Long paragraph — flush current, then split at sentences
            flush()
            for piece in _split_paragraph(block.content, max_tok):
                current.parts.append(piece)
                current.tokens = _token_count(piece)
                current.has_body = True
                flush()
            continue

        if current.tokens + block_tokens > max_tok:
            flush()

        current.parts.append(block.content)
        current.tokens += block_tokens
        current.has_body = True

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Phase 3: Apply overlap
# ---------------------------------------------------------------------------

def _trailing_sentences(text: str, max_tokens: int) -> str:
    """Extract trailing sentences from text up to max_tokens."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    result: list[str] = []
    total = 0
    for sent in reversed(sentences):
        sent_tokens = _token_count(sent)
        if total + sent_tokens > max_tokens:
            break
        result.insert(0, sent)
        total += sent_tokens
    return " ".join(result)


def _apply_overlap(chunks: list[Chunk]) -> list[Chunk]:
    if len(chunks) <= 1:
        return chunks

    overlap_pct = settings.CHUNK_OVERLAP_PERCENT

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        max_overlap_tokens = int(prev.token_count * overlap_pct)
        if max_overlap_tokens < 1:
            continue

        overlap_text = _trailing_sentences(prev.content, max_overlap_tokens)
        if not overlap_text:
            continue

        overlap_tokens = _token_count(overlap_text)
        chunks[i] = Chunk(
            document_id=chunks[i].document_id,
            chunk_index=chunks[i].chunk_index,
            content=overlap_text + "\n\n" + chunks[i].content,
            token_count=chunks[i].token_count + overlap_tokens,
            section_path=chunks[i].section_path,
            has_table=chunks[i].has_table,
            has_code=chunks[i].has_code,
            overlap_tokens=overlap_tokens,
            metadata=chunks[i].metadata,
        )

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(full_markdown: str, document_id: str) -> list[Chunk]:
    """Split markdown into structure-aware chunks. Pure function, no DB calls."""
    blocks = _parse_blocks(full_markdown)
    chunks = _group_blocks(blocks, document_id)
    chunks = _apply_overlap(chunks)
    for chunk in chunks:
        chunk.content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
    logger.info(
        "Chunked document %s → %d chunks (tokens: %s)",
        document_id[:12],
        len(chunks),
        [c.token_count for c in chunks],
    )
    return chunks


async def save_chunks(chunks: list[Chunk]) -> list[str]:
    """Persist chunks to Neon. Idempotent: deletes existing chunks for the document first.

    Returns list of chunk UUIDs.
    """
    if not chunks:
        return []

    document_id = chunks[0].document_id
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM chunks WHERE document_id = $1",
                document_id,
            )

            ids: list[str] = []
            for chunk in chunks:
                row = await conn.fetchrow(
                    "INSERT INTO chunks "
                    "(document_id, chunk_index, content, token_count, "
                    "section_path, has_table, has_code, overlap_tokens, "
                    "metadata, content_hash) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10) "
                    "RETURNING id",
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.token_count,
                    chunk.section_path,
                    chunk.has_table,
                    chunk.has_code,
                    chunk.overlap_tokens,
                    json.dumps(chunk.metadata),
                    chunk.content_hash,
                )
                ids.append(str(row["id"]))

    logger.info("Saved %d chunks for document %s", len(ids), document_id[:12])
    return ids


async def get_chunk_hashes(document_id: str) -> dict[int, str]:
    """Return {chunk_index: content_hash} for a document's chunks."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT chunk_index, content_hash FROM chunks "
        "WHERE document_id = $1::uuid AND content_hash IS NOT NULL",
        document_id,
    )
    return {row["chunk_index"]: row["content_hash"] for row in rows}
