"""Outils partagés entre les agents."""

from typing import List, Dict, Any
from langchain.tools import tool

from src.rag.indexing import build_or_load_index
from src.rag.retrieval import retrieve_context, query_index


# Index global partagé (singleton)
_index = None


def get_index():
    """Retourne l'index global (lazy loading)."""
    global _index
    if _index is None:
        _index = build_or_load_index()
    return _index


@tool
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Recherche des documents pertinents dans le corpus indexé.
    
    Args:
        query: Requête de recherche sémantique
        top_k: Nombre de résultats à retourner (défaut: 5)
    
    Returns:
        Passages pertinents trouvés dans le corpus avec leurs sources
    """
    index = get_index()
    results = retrieve_context(index, query, top_k=top_k)

    if not results:
        return "Aucun document pertinent trouvé pour cette requête."

    output = []
    for i, result in enumerate(results, 1):
        output.append(f"[Résultat {i}] Source: {result['source']} (Score: {result['score']:.3f})\n"
                      f"{result['text']}\n")

    return "\n---\n".join(output)


@tool
def query_knowledge_base(query: str) -> str:
    """
    Interroge la base de connaissances avec une question et retourne une réponse synthétisée.
    
    Args:
        query: Question à poser à la base de connaissances
    
    Returns:
        Réponse synthétisée basée sur les documents indexés
    """
    index = get_index()
    response = query_index(index, query)
    return response


@tool
def analyze_trends(passages: str) -> str:
    """
    Analyse les tendances, méthodologies et approches présentes dans les passages fournis.
    
    Args:
        passages: Passages de texte à analyser
    
    Returns:
        Analyse des tendances identifiées
    """
    return f"Analyse des tendances demandée sur les passages fournis."
