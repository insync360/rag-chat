"""Document parser using LlamaParse agentic_plus tier."""

import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from llama_cloud_services import LlamaParse

from app.config import settings

logger = logging.getLogger(__name__)

AGENTIC_PROMPT = (
    "Extract the document with high fidelity. Preserve all:\n"
    "- Headings and their hierarchy (H1 > H2 > H3)\n"
    "- Tables in structured markdown format with all columns and rows\n"
    "- Code blocks with language tags\n"
    "- Lists (bulleted and numbered) with proper nesting\n"
    "- Image descriptions where images appear\n"
    "- Page numbers and section references\n"
    "Do not summarize or omit any content."
)


class ParsingError(Exception):
    """Raised when document parsing fails."""

    def __init__(self, message: str, filename: str = "", job_id: str = ""):
        self.filename = filename
        self.job_id = job_id
        super().__init__(message)


@dataclass
class ParsedPage:
    page_number: int
    markdown: str


@dataclass
class ParsedDocument:
    filename: str
    content_hash: str
    pages: list[ParsedPage]
    full_markdown: str
    page_count: int
    metadata: dict = field(default_factory=dict)


class LlamaParser:
    """Production document parser using LlamaParse agentic_plus tier."""

    def __init__(self):
        if not settings.LLAMA_CLOUD_API_KEY:
            raise ParsingError("LLAMA_CLOUD_API_KEY is not set")

        self._parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            parse_mode=settings.LLAMAPARSE_MODE,
            model=settings.LLAMAPARSE_MODEL,
            result_type=settings.LLAMAPARSE_RESULT_TYPE,
            high_res_ocr=True,
            adaptive_long_table=True,
            outlined_table_extraction=True,
            output_tables_as_HTML=False,
            parsing_instruction=AGENTIC_PROMPT,
        )

    def _validate_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise ParsingError(f"File not found: {file_path}", filename=file_path.name)

        suffix = file_path.suffix.lower()
        if suffix not in settings.SUPPORTED_FILE_TYPES:
            raise ParsingError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(sorted(settings.SUPPORTED_FILE_TYPES))}",
                filename=file_path.name,
            )

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise ParsingError(
                f"File too large: {size_mb:.1f}MB (max {settings.MAX_FILE_SIZE_MB}MB)",
                filename=file_path.name,
            )

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a document file and return structured markdown output."""
        file_path = Path(file_path)
        self._validate_file(file_path)
        content_hash = self._compute_hash(file_path)

        logger.info("Parsing started: %s (hash=%s)", file_path.name, content_hash[:12])
        start = time.time()

        last_error = None
        for attempt in range(1, settings.PARSE_MAX_RETRIES + 1):
            try:
                result = await self._parser.aparse(str(file_path))
                break
            except Exception as exc:
                last_error = exc
                if attempt < settings.PARSE_MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning(
                        "Parse attempt %d/%d failed for %s, retrying in %ds: %s",
                        attempt, settings.PARSE_MAX_RETRIES, file_path.name, wait, exc,
                    )
                    import asyncio
                    await asyncio.sleep(wait)
        else:
            raise ParsingError(
                f"Parsing failed after {settings.PARSE_MAX_RETRIES} attempts: {last_error}",
                filename=file_path.name,
            )

        nodes = result.get_markdown_nodes(split_by_page=True)

        pages = [
            ParsedPage(page_number=i + 1, markdown=node.text)
            for i, node in enumerate(nodes)
        ]
        full_markdown = "\n\n".join(page.markdown for page in pages)
        elapsed = time.time() - start

        logger.info(
            "Parsing complete: %s — %d pages in %.1fs",
            file_path.name, len(pages), elapsed,
        )

        return ParsedDocument(
            filename=file_path.name,
            content_hash=content_hash,
            pages=pages,
            full_markdown=full_markdown,
            page_count=len(pages),
            metadata={
                "tier": "agentic_plus",
                "model": settings.LLAMAPARSE_MODEL,
                "processing_time_seconds": round(elapsed, 2),
            },
        )

    async def parse_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """Parse raw bytes (e.g. from an upload endpoint)."""
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            return await self.parse(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
