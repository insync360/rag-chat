import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
    LLAMA_CLOUD_API_KEY: str = os.environ.get("LLAMA_CLOUD_API_KEY", "")

    LLAMAPARSE_MODE: str = "parse_page_with_agent"  # agentic_plus tier
    LLAMAPARSE_MODEL: str = "anthropic-sonnet-4.0"
    LLAMAPARSE_RESULT_TYPE: str = "markdown"

    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    ENRICHMENT_MODEL: str = "gpt-4o-mini"
    ENRICHMENT_CONCURRENCY: int = 5

    SUPPORTED_FILE_TYPES: set[str] = {".pdf", ".docx", ".pptx", ".txt", ".md", ".html"}
    MAX_FILE_SIZE_MB: int = 50
    PARSE_TIMEOUT_SECONDS: int = 300  # 5 minutes
    PARSE_MAX_RETRIES: int = 3

    CHUNK_MIN_TOKENS: int = 256
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_PERCENT: float = 0.12  # 12% overlap

    # Neo4j
    NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "")

    # Graph extraction
    GRAPH_EXTRACTION_MODEL: str = "gpt-4o"
    GRAPH_EXTRACTION_CONCURRENCY: int = 3
    GRAPH_EXTRACTION_ENABLED: bool = True
    GRAPH_BATCH_SIZE: int = 100


settings = Settings()
