"""Configuration globale du projet WAC Sport Analytics."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration du systeme multi-agents WAC."""

    # LLM Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # RAG Configuration
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/processed/vector_store")
    data_path: str = os.getenv("DATA_PATH", "./data/raw")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    def __post_init__(self):
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)


config = Config()
