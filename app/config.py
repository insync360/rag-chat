import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
    LLAMA_CLOUD_API_KEY: str = os.environ.get("LLAMA_CLOUD_API_KEY", "")

    LLAMAPARSE_MODE: str = "parse_page_with_agent"  # agentic_plus tier
    LLAMAPARSE_MODEL: str = "anthropic-sonnet-4.0"
    LLAMAPARSE_RESULT_TYPE: str = "markdown"

    SUPPORTED_FILE_TYPES: set[str] = {".pdf", ".docx", ".pptx", ".txt", ".md", ".html"}
    MAX_FILE_SIZE_MB: int = 50
    PARSE_TIMEOUT_SECONDS: int = 300  # 5 minutes
    PARSE_MAX_RETRIES: int = 3

    CHUNK_MIN_TOKENS: int = 256
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_PERCENT: float = 0.12  # 12% overlap


settings = Settings()
