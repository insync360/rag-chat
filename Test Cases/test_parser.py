"""Quick test: parse a document with LlamaParse agentic_plus.

Usage:
    python test_parser.py <file_path>

Requires LLAMA_CLOUD_API_KEY in .env
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


async def main():
    from app.ingestion.parser import LlamaParser

    path = sys.argv[1] if len(sys.argv) > 1 else "production_rag_blueprint (1).pdf"
    parser = LlamaParser()
    doc = await parser.parse(path)

    print(f"\n{'='*60}")
    print(f"Filename:     {doc.filename}")
    print(f"Content hash: {doc.content_hash}")
    print(f"Pages:        {doc.page_count}")
    print(f"Metadata:     {doc.metadata}")
    print(f"{'='*60}")
    print(f"\n--- First 500 chars of markdown ---")
    print(doc.full_markdown[:500])


if __name__ == "__main__":
    asyncio.run(main())
