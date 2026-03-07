"""Document version tracker — content_hash deduplication, deprecate never delete."""

import logging
from dataclasses import dataclass
from datetime import datetime

import asyncpg

from app.database import get_pool
from app.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    id: str
    filename: str
    content_hash: str
    version: int
    status: str
    page_count: int
    ingested_at: datetime
    metadata: dict


class VersionTracker:
    """Tracks document versions in Neon PostgreSQL.

    Rules:
    - Same content_hash → skip (idempotent), return existing record
    - Same filename, new hash → deprecate old versions, insert as next version
    - New filename → insert as version 1
    """

    async def track(self, doc: ParsedDocument) -> tuple[DocumentRecord, bool]:
        """Register a parsed document. Returns (record, is_new).

        is_new=False means the exact content was already ingested (cache hit).
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            # 1. Idempotency: check if this exact content already exists
            existing = await conn.fetchrow(
                "SELECT id, filename, content_hash, version, status, "
                "page_count, ingested_at, metadata "
                "FROM documents WHERE content_hash = $1",
                doc.content_hash,
            )
            if existing:
                record = self._row_to_record(existing)
                logger.info(
                    "Duplicate detected: %s (hash=%s) → existing version %d",
                    doc.filename, doc.content_hash[:12], record.version,
                )
                return record, False

            # 2. Determine next version number for this filename
            async with conn.transaction():
                max_version = await conn.fetchval(
                    "SELECT COALESCE(MAX(version), 0) FROM documents "
                    "WHERE filename = $1",
                    doc.filename,
                )
                new_version = max_version + 1

                # 3. Deprecate all active versions of this filename
                deprecated_count = await conn.execute(
                    "UPDATE documents SET status = 'deprecated', deprecated_at = now() "
                    "WHERE filename = $1 AND status = 'active'",
                    doc.filename,
                )
                if "UPDATE" in deprecated_count and not deprecated_count.endswith("0"):
                    logger.info(
                        "Deprecated previous versions of %s (%s)",
                        doc.filename, deprecated_count,
                    )

                # 4. Insert new active version
                import json
                row = await conn.fetchrow(
                    "INSERT INTO documents "
                    "(filename, content_hash, version, status, page_count, "
                    "full_markdown, metadata) "
                    "VALUES ($1, $2, $3, 'active', $4, $5, $6::jsonb) "
                    "RETURNING id, filename, content_hash, version, status, "
                    "page_count, ingested_at, metadata",
                    doc.filename,
                    doc.content_hash,
                    new_version,
                    doc.page_count,
                    doc.full_markdown,
                    json.dumps(doc.metadata),
                )

            record = self._row_to_record(row)
            logger.info(
                "Ingested: %s v%d (hash=%s, pages=%d)",
                record.filename, record.version, record.content_hash[:12], record.page_count,
            )
            return record, True

    async def get_active(self, filename: str) -> DocumentRecord | None:
        """Get the current active version of a document."""
        pool = await get_pool()
        row = await pool.fetchrow(
            "SELECT id, filename, content_hash, version, status, "
            "page_count, ingested_at, metadata "
            "FROM documents WHERE filename = $1 AND status = 'active'",
            filename,
        )
        return self._row_to_record(row) if row else None

    async def get_history(self, filename: str) -> list[DocumentRecord]:
        """Get all versions of a document, newest first."""
        pool = await get_pool()
        rows = await pool.fetch(
            "SELECT id, filename, content_hash, version, status, "
            "page_count, ingested_at, metadata "
            "FROM documents WHERE filename = $1 ORDER BY version DESC",
            filename,
        )
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: asyncpg.Record) -> DocumentRecord:
        import json
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return DocumentRecord(
            id=str(row["id"]),
            filename=row["filename"],
            content_hash=row["content_hash"],
            version=row["version"],
            status=row["status"],
            page_count=row["page_count"],
            ingested_at=row["ingested_at"],
            metadata=meta if meta else {},
        )
