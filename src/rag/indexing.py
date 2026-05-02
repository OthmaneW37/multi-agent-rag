"""Module d'indexation pour le pipeline RAG avec LlamaIndex."""

from typing import List

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
import chromadb

from src.config import config
from src.rag.ingestion import load_documents


def get_embedding_model():
    """Retourne le modele d'embedding configure."""
    return HuggingFaceEmbedding(model_name=config.embedding_model)


def get_llm_for_llamaindex():
    """Retourne le LLM Ollama configure pour LlamaIndex."""
    return Ollama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        request_timeout=120.0,
    )


def create_index(documents: List) -> VectorStoreIndex:
    """
    Cree un index vectoriel a partir des documents.

    Args:
        documents: Liste de documents LlamaIndex

    Returns:
        VectorStoreIndex: L'index cree
    """
    # Configuration des embeddings et LLM
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    Settings.llm = get_llm_for_llamaindex()

    # Configuration du parser de chunks
    node_parser = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # Configuration du vector store ChromaDB
    chroma_client = chromadb.PersistentClient(path=config.vector_store_path)
    chroma_collection = chroma_client.get_or_create_collection("wac_sport_data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Creation de l'index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        node_parser=node_parser,
    )

    print(f"[RAG] Index cree avec {len(documents)} documents.")
    print(f"[RAG] Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    print(f"[RAG] Embedding model: {config.embedding_model}")

    return index


def load_index() -> VectorStoreIndex:
    """Charge un index existant depuis le stockage persistant."""
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    Settings.llm = get_llm_for_llamaindex()

    chroma_client = chromadb.PersistentClient(path=config.vector_store_path)
    chroma_collection = chroma_client.get_or_create_collection("wac_sport_data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    print("[RAG] Index charge depuis le stockage persistant.")
    return index


def index_exists() -> bool:
    """Verifie si un index existe deja dans le vector store."""
    try:
        chroma_client = chromadb.PersistentClient(path=config.vector_store_path)
        collection = chroma_client.get_collection("wac_sport_data")
        return collection.count() > 0
    except Exception:
        return False


def build_or_load_index(
    data_path: str = None, force_rebuild: bool = False
) -> VectorStoreIndex:
    """
    Construit un nouvel index ou charge un index existant.

    Args:
        data_path: Chemin vers les documents (defaut: config.data_path)
        force_rebuild: Force la reconstruction de l'index

    Returns:
        VectorStoreIndex: L'index pret a l'emploi
    """
    data_path = data_path or config.data_path

    if not force_rebuild and index_exists():
        print("[RAG] Index existant detecte. Chargement...")
        return load_index()

    import os

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    if files:
        print(f"[RAG] Documents trouves dans {data_path}. Creation d'un nouvel index...")
        documents = load_documents(data_path)
        if documents:
            return create_index(documents)

    print("[RAG] Aucun document trouve. Chargement de l'index existant...")
    return load_index()
