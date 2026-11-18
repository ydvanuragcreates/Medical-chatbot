"""Application settings, read once from environment variables (and .env)."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(override=True)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    pinecone_api_key: str | None
    pinecone_index: str
    openai_model: str
    temperature: float
    embedding_model: str
    top_k: int
    max_history_messages: int
    log_level: str

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        pinecone_api_key=(os.getenv("PINECONE_API_KEY") or "").strip() or None,
        pinecone_index=os.getenv("PINECONE_INDEX", "medicalbot"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=_float("OPENAI_TEMPERATURE", 0.3),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        top_k=_int("RETRIEVAL_TOP_K", 4),
        max_history_messages=_int("MAX_HISTORY_MESSAGES", 8),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
