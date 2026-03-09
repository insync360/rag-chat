def __getattr__(name: str):
    """Lazy imports to avoid circular import with app.graph."""
    if name in ("ingest_file", "ingest_files"):
        from app.ingestion.pipeline import ingest_file, ingest_files
        return {"ingest_file": ingest_file, "ingest_files": ingest_files}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ingest_file", "ingest_files"]
