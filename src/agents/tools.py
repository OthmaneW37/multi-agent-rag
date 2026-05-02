"""Outils partages entre les agents du systeme WAC Sport Analytics."""

from typing import List, Dict, Any
from langchain.tools import tool

from src.rag.indexing import build_or_load_index
from src.rag.retrieval import retrieve_context, query_index


# Index global partage (singleton)
_index = None

# Seuil minimum de pertinence pour eviter les hallucinations.
# Le modele all-MiniLM-L6-v2 produit des scores cosine relativement bas
# (0.20-0.35 typiquement). 0.15 est un bon compromis.
MIN_RELEVANCE_SCORE = 0.15


def get_index():
    """Retourne l'index global (lazy loading)."""
    global _index
    if _index is None:
        _index = build_or_load_index()
    return _index


def _filter_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filtre les resultats par score de pertinence minimum."""
    filtered = [r for r in results if r.get("score", 0) >= MIN_RELEVANCE_SCORE]
    return filtered


def _format_results(results: List[Dict[str, Any]]) -> str:
    """Formate les resultats pour l'agent."""
    output = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "inconnu")
        score = result.get("score", 0)
        output.append(
            f"[Resultat {i}] Source: {source} (Score: {score:.3f})\n"
            f"{result['text']}\n"
        )
    return "\n---\n".join(output)


@tool
def search_match_reports(query: str, top_k: int = 5) -> str:
    """
    Recherche des rapports de matchs et statistiques dans la base de donnees.

    Args:
        query: Requete de recherche semantique (ex: "WAC vs Raja resultat", "forme WAC domicile")
        top_k: Nombre de resultats a retourner (defaut: 5)

    Returns:
        Passages pertinents trouves dans le corpus avec leurs sources.
        Si aucun document n'est suffisamment pertinent, retourne un message
        indiquant que les donnees sont insuffisantes.
    """
    index = get_index()
    results = retrieve_context(index, query, top_k=top_k)
    filtered = _filter_results(results)

    if not filtered:
        return (
            "[ATTENTION] Aucun document suffisamment pertinent trouve dans la base "
            "de donnees pour cette requete. Les donnees disponibles ne permettent pas "
            "de repondre de maniere fiable."
        )

    return _format_results(filtered)


@tool
def query_sport_database(query: str) -> str:
    """
    Interroge la base de connaissances sportives avec une question.

    Args:
        query: Question a poser (ex: "Quel est le bilan du WAC a domicile cette saison?")

    Returns:
        Reponse synthetisee basee UNIQUEMENT sur les documents indexes.
        Si les documents ne contiennent pas assez d'informations, indique
        clairement que la reponse n'est pas disponible dans la base.
    """
    index = get_index()
    response = query_index(index, query)
    response_str = str(response)

    # Guardrail : si la reponse est vide ou trop courte, c'est probablement
    # que le query engine n'a pas trouve de documents pertinents
    if len(response_str.strip()) < 20:
        return (
            "[ATTENTION] La base de donnees ne contient pas suffisamment "
            "d'informations pour repondre a cette question de maniere fiable."
        )

    return response_str


@tool
def get_player_stats(player_name: str) -> str:
    """
    Recupere les statistiques d'un joueur specifique depuis la base.

    Args:
        player_name: Nom du joueur (ex: "Yahya Jabrane", "Zouhair El Moutaraji")

    Returns:
        Statistiques du joueur trouvees dans la base. Si aucune statistique
        suffisamment pertinente n'est trouvee, indique que le joueur n'est pas
        reference ou que les donnees sont insuffisantes.
    """
    index = get_index()
    results = retrieve_context(index, f"statistiques joueur {player_name}", top_k=3)
    filtered = _filter_results(results)

    if not filtered:
        return (
            f"[ATTENTION] Aucune statistique suffisamment pertinente trouvee "
            f"pour le joueur '{player_name}' dans la base de donnees."
        )

    output = f"Statistiques de {player_name}:\n"
    for result in filtered:
        output += f"\nSource: {result.get('source', 'inconnu')}\n{result['text']}\n"

    return output
