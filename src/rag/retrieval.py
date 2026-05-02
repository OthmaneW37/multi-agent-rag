"""Module de retrieval pour le pipeline RAG."""

from typing import List, Dict, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


def create_retriever(index: VectorStoreIndex, similarity_top_k: int = 5):
    """
    Cree un retriever pour l'index donne.

    Args:
        index: L'index vectoriel
        similarity_top_k: Nombre de resultats a retourner

    Returns:
        VectorIndexRetriever: Le retriever configure
    """
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )


def create_query_engine(index: VectorStoreIndex, similarity_top_k: int = 5):
    """
    Cree un moteur de requete complet.

    Args:
        index: L'index vectoriel
        similarity_top_k: Nombre de resultats a retourner

    Returns:
        RetrieverQueryEngine: Le moteur de requete
    """
    retriever = create_retriever(index, similarity_top_k)

    # Post-processeur pour filtrer par similarite minimale
    # Le modele all-MiniLM-L6-v2 produit des scores cosine relativement bas
    # (typiquement 0.20-0.35 pour des documents texte). Un seuil de 0.15
    # elimine le bruit sans supprimer les documents pertinents.
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.15)
    ]

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
        response_mode="compact",
    )


def retrieve_context(
    index: VectorStoreIndex, query: str, top_k: int = 5, min_score: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Recupere le contexte pertinent pour une requete.

    Args:
        index: L'index vectoriel
        query: La requete de recherche
        top_k: Nombre de passages a retourner
        min_score: Score de similarite minimum pour accepter un resultat

    Returns:
        Liste de dictionnaires contenant le texte et les metadata
    """
    retriever = create_retriever(index, similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        score = getattr(node, "score", 0)
        if score is not None and score < min_score:
            continue
        results.append({
            "text": node.node.text,
            "score": score,
            "metadata": node.node.metadata,
            "source": node.node.metadata.get("source", "unknown"),
        })

    return results


def query_index(index: VectorStoreIndex, query: str) -> str:
    """
    Interroge l'index avec une requete et retourne les passages pertinents.

    NOTE : Cette fonction ne fait PAS appel au LLM pour synthetiser.
    Elle retourne directement les passages trouves dans l'index.
    Cela evite les timeouts avec Ollama et les hallucinations.

    Args:
        index: L'index vectoriel
        query: La requete

    Returns:
        Les passages pertinents concatenes avec leurs sources
    """
    results = retrieve_context(index, query, top_k=5, min_score=0.15)

    if not results:
        return (
            "Aucun document suffisamment pertinent trouve dans la base "
            "de donnees pour cette requete."
        )

    lines = [f"Requete : {query}", "=" * 50, ""]
    for i, result in enumerate(results, 1):
        lines.append(
            f"[Source {i}] {result.get('source', 'inconnu')} "
            f"(pertinence: {result.get('score', 0):.3f})"
        )
        lines.append(result["text"])
        lines.append("")

    return "\n".join(lines)
