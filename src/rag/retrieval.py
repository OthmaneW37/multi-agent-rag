"""Module de retrieval pour le pipeline RAG."""

from typing import List, Dict, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from src.config import config


def create_retriever(index: VectorStoreIndex, similarity_top_k: int = 5):
    """
    Crée un retriever pour l'index donné.
    
    Args:
        index: L'index vectoriel
        similarity_top_k: Nombre de résultats à retourner
    
    Returns:
        VectorIndexRetriever: Le retriever configuré
    """
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )
    return retriever


def create_query_engine(index: VectorStoreIndex, similarity_top_k: int = 5):
    """
    Crée un moteur de requête complet.
    
    Args:
        index: L'index vectoriel
        similarity_top_k: Nombre de résultats à retourner
    
    Returns:
        RetrieverQueryEngine: Le moteur de requête
    """
    retriever = create_retriever(index, similarity_top_k)

    # Post-processeur pour filtrer par similarite minimale
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.3)
    ]

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
        response_mode="compact",
    )

    return query_engine


def retrieve_context(index: VectorStoreIndex, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Récupère le contexte pertinent pour une requête.
    
    Args:
        index: L'index vectoriel
        query: La requête de recherche
        top_k: Nombre de passages à retourner
    
    Returns:
        Liste de dictionnaires contenant le texte et les métadonnées
    """
    retriever = create_retriever(index, similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        results.append({
            "text": node.node.text,
            "score": node.score,
            "metadata": node.node.metadata,
            "source": node.node.metadata.get("source", "unknown"),
        })

    return results


def query_index(index: VectorStoreIndex, query: str) -> str:
    """
    Interroge l'index avec une requête et retourne une réponse synthétisée.
    
    Args:
        index: L'index vectoriel
        query: La requête
    
    Returns:
        La réponse synthétisée
    """
    query_engine = create_query_engine(index)
    response = query_engine.query(query)
    return str(response)
