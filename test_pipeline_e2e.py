"""End-to-end pipeline test: parse → version track → chunk → enrich → save.

Run: .venv/Scripts/python test_pipeline_e2e.py
"""

import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PDF_PATH = r"C:\ganesh\code\rag_chat\SAE-Steer_DSE316_COURSE_PROJECT_PROPOSAL.pdf"


async def run_pipeline():
    from app.database import get_pool, close_pool
    from app.ingestion.parser import LlamaParser
    from app.ingestion.version_tracker import VersionTracker
    from app.ingestion.chunker import chunk_document, save_chunks
    from app.ingestion.enricher import enrich_chunks

    # ── Step 1: Parse ──
    logger.info("=" * 60)
    logger.info("STEP 1: Parsing PDF with LlamaParse")
    logger.info("=" * 60)
    parser = LlamaParser()
    doc = await parser.parse(PDF_PATH)
    logger.info("Parsed: %s — %d pages, %d chars, hash=%s",
                doc.filename, doc.page_count, len(doc.full_markdown), doc.content_hash[:12])

    # ── Step 2: Version Track ──
    logger.info("=" * 60)
    logger.info("STEP 2: Version tracking")
    logger.info("=" * 60)
    tracker = VersionTracker()
    record, is_new = await tracker.track(doc)
    logger.info("Document record: id=%s, version=%d, is_new=%s", record.id[:12], record.version, is_new)

    if not is_new:
        logger.info("Document already ingested (same content_hash). Continuing with existing record.")

    # ── Step 3: Chunk ──
    logger.info("=" * 60)
    logger.info("STEP 3: Structure-aware chunking")
    logger.info("=" * 60)
    chunks = chunk_document(doc.full_markdown, record.id)
    logger.info("Produced %d chunks", len(chunks))
    for i, c in enumerate(chunks):
        logger.info("  Chunk %d: %d tokens | table=%s code=%s | path='%s' | preview='%s'",
                     i, c.token_count, c.has_table, c.has_code,
                     c.section_path, c.content[:80].replace("\n", " "))

    # ── Step 4: Enrich ──
    logger.info("=" * 60)
    logger.info("STEP 4: Metadata enrichment (GPT-4o-mini)")
    logger.info("=" * 60)
    chunks = await enrich_chunks(chunks, record)
    for i, c in enumerate(chunks):
        m = c.metadata
        logger.info("  Chunk %d metadata:", i)
        logger.info("    chunk_type:    %s", m.get("chunk_type"))
        logger.info("    freshness:     %s", m.get("freshness_score"))
        logger.info("    summary:       %s", (m.get("summary", "")[:100] + "...") if m.get("summary") else "(none)")
        logger.info("    keywords:      %s", m.get("keywords", []))
        logger.info("    questions:     %s", m.get("hypothetical_questions", [])[:2])
        logger.info("    entities:      %s", m.get("entities", {}))

    # ── Step 5: Save ──
    logger.info("=" * 60)
    logger.info("STEP 5: Saving chunks to Neon")
    logger.info("=" * 60)
    chunk_ids = await save_chunks(chunks)
    logger.info("Saved %d chunks: %s", len(chunk_ids), chunk_ids)

    # ── Step 6: Verify round-trip ──
    logger.info("=" * 60)
    logger.info("STEP 6: Verifying DB round-trip")
    logger.info("=" * 60)
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, chunk_index, token_count, section_path, has_table, has_code, metadata "
        "FROM chunks WHERE document_id = $1 ORDER BY chunk_index",
        record.id,
    )
    logger.info("Retrieved %d chunks from DB", len(rows))

    errors = []
    for row in rows:
        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
        idx = row["chunk_index"]

        # Validate required fields
        for field in ("chunk_type", "freshness_score", "document_id", "version", "enriched_at", "ingested_at"):
            if field not in meta:
                errors.append(f"Chunk {idx}: missing '{field}'")

        # Validate LLM fields for non-tiny chunks
        if row["token_count"] >= 10:
            for field in ("summary", "keywords", "hypothetical_questions", "entities"):
                if field not in meta:
                    errors.append(f"Chunk {idx}: missing LLM field '{field}'")

        logger.info("  Chunk %d: %d tokens | type=%s | keys=%s",
                     idx, row["token_count"], meta.get("chunk_type"), sorted(meta.keys()))

    if errors:
        logger.error("VALIDATION ERRORS:")
        for e in errors:
            logger.error("  %s", e)
        sys.exit(1)
    else:
        logger.info("ALL VALIDATIONS PASSED")

    await close_pool()

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("  Document: %s", doc.filename)
    logger.info("  Pages:    %d", doc.page_count)
    logger.info("  Chunks:   %d", len(chunks))
    logger.info("  Enriched: %d/%d with LLM metadata", sum(1 for c in chunks if c.metadata.get("summary")), len(chunks))
    logger.info("  DB IDs:   %s", chunk_ids)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline())
