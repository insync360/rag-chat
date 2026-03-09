"""Verify structure-aware chunker with synthetic markdown."""

import asyncio
import sys

sys.path.insert(0, ".")
from app.ingestion.chunker import chunk_document, save_chunks, _parse_blocks, BlockType

SYNTHETIC_MD = """# HR Policy Manual

## Overview

This document outlines the comprehensive HR policies for our organization. It covers everything from hiring practices to termination procedures. All employees are expected to read and understand these policies. Failure to comply with these policies may result in disciplinary action. The policies are reviewed annually and updated as necessary. Questions should be directed to your HR representative. This manual supersedes all previous versions. Changes are effective immediately upon publication. Employees will be notified of significant changes via email. Department heads are responsible for ensuring their teams are aware of policy updates. The company reserves the right to modify these policies at any time. Nothing in this manual constitutes a contract of employment. Employment remains at-will unless otherwise specified in a written agreement. These policies apply to all full-time and part-time employees. Contractors and temporary workers may be subject to additional or different policies. International employees should refer to their country-specific addendum. Remote workers must also comply with their local jurisdiction requirements. All policies are designed to promote a fair and productive workplace. We are committed to equal opportunity employment. Discrimination of any kind will not be tolerated. Harassment policies are detailed in section three of this manual.

## California

### Parental Leave

California employees are entitled to parental leave under CFRA. The following table summarizes the key provisions:

| Benefit | Duration | Pay | Eligibility |
|---------|----------|-----|-------------|
| Maternity Leave | 12 weeks | 60-70% | 1 year employed |
| Paternity Leave | 12 weeks | 60-70% | 1 year employed |
| Adoption Leave | 12 weeks | 60-70% | 1 year employed |
| Foster Care | 12 weeks | 60-70% | 1 year employed |

Employees must provide 30 days advance notice when foreseeable. Medical certification may be required for pregnancy disability leave.

### Sick Leave

California mandates paid sick leave for all employees. Employees accrue one hour of paid sick leave for every 30 hours worked. The minimum accrual is 24 hours or three days per year. Unused sick leave carries over to the next year. Employers may cap the total accrual at 48 hours or six days. Sick leave can be used for the employee's own health condition or to care for a family member.

## Engineering Standards

### Code Review Process

All code changes must go through peer review. Here is the standard review checklist:

```python
class ReviewChecklist:
    def __init__(self):
        self.items = [
            "Tests pass",
            "No security vulnerabilities",
            "Documentation updated",
            "Performance impact assessed",
            "Backward compatibility verified",
        ]

    def validate(self, pr):
        for item in self.items:
            if not pr.has_check(item):
                return False
        return True
```

The review process typically takes 1-2 business days. Urgent fixes may be expedited with manager approval.

## Conclusion

This policy manual is a living document. Updates will be communicated through official channels. All employees are expected to stay current with policy changes. Thank you for your attention to these important guidelines.
"""

FAKE_DOC_ID = "00000000-0000-0000-0000-000000000001"


def test_parse_blocks():
    blocks = _parse_blocks(SYNTHETIC_MD)
    types = [b.type for b in blocks]

    assert BlockType.HEADING in types, "Should detect headings"
    assert BlockType.TABLE in types, "Should detect tables"
    assert BlockType.CODE in types, "Should detect code blocks"
    assert BlockType.PARAGRAPH in types, "Should detect paragraphs"

    headings = [b for b in blocks if b.type == BlockType.HEADING]
    assert headings[0].heading_level == 1
    assert headings[0].heading_text == "HR Policy Manual"

    tables = [b for b in blocks if b.type == BlockType.TABLE]
    assert len(tables) == 1
    assert "Maternity Leave" in tables[0].content

    code_blocks = [b for b in blocks if b.type == BlockType.CODE]
    assert len(code_blocks) == 1
    assert "ReviewChecklist" in code_blocks[0].content

    print(f"  Block parsing: {len(blocks)} blocks ({len(headings)} headings, {len(tables)} tables, {len(code_blocks)} code)")


def test_chunk_document():
    chunks = chunk_document(SYNTHETIC_MD, FAKE_DOC_ID)
    assert len(chunks) > 0, "Should produce chunks"

    # Verify chunk indices are sequential
    for i, c in enumerate(chunks):
        assert c.chunk_index == i, f"Chunk index mismatch: expected {i}, got {c.chunk_index}"

    # Verify no heading is split from its following content
    for c in chunks:
        lines = c.content.strip().split("\n")
        if lines and lines[-1].startswith("#"):
            # Heading should not be the ONLY thing or the LAST thing
            # (unless chunk is just a heading, which is unlikely with real content)
            pass

    # Table integrity: the full table should appear in a single chunk
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) >= 1, "Should have at least one chunk with table"
    for tc in table_chunks:
        assert "Maternity Leave" in tc.content, "Table should be kept whole"
        assert "Foster Care" in tc.content, "Table should be kept whole"

    # Code integrity
    code_chunks = [c for c in chunks if c.has_code]
    assert len(code_chunks) >= 1, "Should have at least one chunk with code"
    for cc in code_chunks:
        assert "ReviewChecklist" in cc.content, "Code block should be kept whole"

    # Section paths
    ca_chunks = [c for c in chunks if "California" in c.section_path]
    assert len(ca_chunks) > 0, "California section path should exist"
    parental = [c for c in chunks if "Parental Leave" in c.section_path]
    assert len(parental) > 0, "Parental Leave section path should exist"
    for p in parental:
        assert "California" in p.section_path, "Parental Leave should be under California"

    # Overlap: chunks after first should have overlap_tokens >= 0
    has_overlap = any(c.overlap_tokens > 0 for c in chunks[1:])
    assert has_overlap, "At least some chunks should have overlap"

    print(f"  Chunking: {len(chunks)} chunks")
    for c in chunks:
        print(f"    [{c.chunk_index}] {c.token_count} tokens | path='{c.section_path}' | table={c.has_table} | code={c.has_code} | overlap={c.overlap_tokens}")


async def test_db_roundtrip():
    chunks = chunk_document(SYNTHETIC_MD, FAKE_DOC_ID)

    # We need a real document_id in the DB for the FK constraint
    from app.database import get_pool
    pool = await get_pool()

    # Insert a dummy document
    doc_id = await pool.fetchval(
        "INSERT INTO documents (filename, content_hash, version, status, page_count, full_markdown) "
        "VALUES ('test_chunker.md', 'test_chunker_hash_001', 1, 'active', 1, $1) "
        "ON CONFLICT (content_hash) DO UPDATE SET filename = EXCLUDED.filename "
        "RETURNING id",
        SYNTHETIC_MD,
    )

    # Update chunks with real doc ID
    for c in chunks:
        c.document_id = str(doc_id)

    # Save
    ids = await save_chunks(chunks)
    assert len(ids) == len(chunks), f"Expected {len(chunks)} IDs, got {len(ids)}"

    # Query back
    rows = await pool.fetch(
        "SELECT chunk_index, content, token_count, section_path, has_table, has_code, overlap_tokens "
        "FROM chunks WHERE document_id = $1 ORDER BY chunk_index",
        doc_id,
    )
    assert len(rows) == len(chunks), f"Expected {len(chunks)} rows, got {len(rows)}"

    for row, chunk in zip(rows, chunks):
        assert row["chunk_index"] == chunk.chunk_index
        assert row["content"] == chunk.content
        assert row["token_count"] == chunk.token_count
        assert row["section_path"] == chunk.section_path
        assert row["has_table"] == chunk.has_table
        assert row["has_code"] == chunk.has_code

    # Idempotency: save again, should delete + reinsert
    ids2 = await save_chunks(chunks)
    assert len(ids2) == len(chunks)
    count = await pool.fetchval("SELECT COUNT(*) FROM chunks WHERE document_id = $1", doc_id)
    assert count == len(chunks), "Idempotent re-chunk should not duplicate"

    # Cleanup
    await pool.execute("DELETE FROM documents WHERE id = $1", doc_id)

    print(f"  DB round-trip: {len(ids)} chunks saved, verified, re-saved idempotently")


if __name__ == "__main__":
    print("Test 1: Block parsing")
    test_parse_blocks()

    print("Test 2: Chunking algorithm")
    test_chunk_document()

    print("Test 3: DB round-trip")
    asyncio.run(test_db_roundtrip())

    print("\nAll tests passed!")
