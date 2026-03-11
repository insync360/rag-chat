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
    COREF_ENABLED: bool = True

    # Incremental graph updates
    INCREMENTAL_GRAPH_ENABLED: bool = True
    ENTITY_FUZZY_THRESHOLD: float = 85.0        # rapidfuzz token_sort_ratio (0-100)
    ENTITY_EMBEDDING_THRESHOLD: float = 0.92    # cosine similarity (0-1)
    ENTITY_EMBEDDING_MODEL: str = "text-embedding-3-large"
    ENTITY_EMBEDDING_DIMENSIONS: int = 256      # reduced dims — entity names are short

    # Graph embeddings (GraphSAGE)
    GRAPH_EMBEDDINGS_ENABLED: bool = True
    GRAPHSAGE_INPUT_DIM: int = 256        # OpenAI initial features dim
    GRAPHSAGE_HIDDEN_DIM: int = 128
    GRAPHSAGE_OUTPUT_DIM: int = 128       # structural embedding dim
    GRAPHSAGE_NEIGHBOR_SAMPLES: int = 25  # K neighbors per node per layer
    GRAPHSAGE_EPOCHS: int = 200
    GRAPHSAGE_LR: float = 0.01
    GRAPHSAGE_NEG_RATIO: int = 5         # negative samples per positive edge
    GRAPHSAGE_BATCH_SIZE: int = 512       # edge batch size for training
    GRAPHSAGE_SEED: int = 42
    GRAPHSAGE_MODEL_DIR: str = "models"

    # TransE relation embeddings
    TRANSE_ENABLED: bool = True
    TRANSE_DIM: int = 128
    TRANSE_EPOCHS: int = 200
    TRANSE_LR: float = 0.01
    TRANSE_MARGIN: float = 1.0
    TRANSE_BATCH_SIZE: int = 512
    TRANSE_SEED: int = 42
    TRANSE_MODEL_DIR: str = "models"

    # Chunk embeddings
    CHUNK_EMBEDDING_MODEL: str = "text-embedding-3-large"
    CHUNK_EMBEDDING_DIMENSIONS: int = 2000  # Neon pgvector HNSW limit (text-embedding-3-large supports reduced dims)
    CHUNK_EMBEDDING_BATCH_SIZE: int = 2048

    # Community detection
    COMMUNITY_DETECTION_ENABLED: bool = True
    COMMUNITY_SUMMARY_ENABLED: bool = True
    COMMUNITY_SUMMARY_MODEL: str = "gpt-4o-mini"
    COMMUNITY_MIN_SIZE: int = 2          # skip singletons for summaries
    COMMUNITY_RESOLUTION: float = 1.0   # Leiden resolution (higher = more communities)

    # Community summary embeddings
    COMMUNITY_SUMMARY_EMBEDDING_ENABLED: bool = True
    COMMUNITY_SUMMARY_EMBEDDING_MODEL: str = "text-embedding-3-large"
    COMMUNITY_SUMMARY_EMBEDDING_DIMENSIONS: int = 512
    COMMUNITY_SUMMARY_EMBEDDING_BATCH_SIZE: int = 2048

    # Hybrid chunk-entity embeddings
    HYBRID_CHUNK_ENTITY_ENABLED: bool = True
    HYBRID_CHUNK_TEXT_DIM: int = 512  # MRL truncation of 2000-dim chunk embedding


settings = Settings()
