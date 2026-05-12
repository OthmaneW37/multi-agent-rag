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
    """Formate les resultats pour l'agent (tronques a 300 caractères pour limiter les tokens)."""
    output = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "inconnu")
        score = result.get("score", 0)
        text = result["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        output.append(
            f"[R{i}] {source} (s:{score:.2f})\n{text}\n"
        )
    return "\n---\n".join(output)


@tool
def search_match_reports(query: str, top_k: int = 3) -> str:
    """Recherche semantique dans les stats FootyStats Botola Pro 2025/2026.
    Args: query (ex: "WAC vs Raja resultat", "forme WAC domicile"), top_k (defaut 3)
    Retourne passages pertinents avec sources, ou message si donnees insuffisantes."""
    index = get_index()
    results = retrieve_context(index, query, top_k=top_k)
    filtered = _filter_results(results)
    if not filtered:
        return "[ATTENTION] Aucun document pertinent trouve pour cette requete."
    return _format_results(filtered)


@tool
def query_sport_database(query: str) -> str:
    """Interroge la base FootyStats avec une question precise.
    Args: query (ex: "bilan WAC domicile"). Retourne reponse basee sur les documents indexes."""
    index = get_index()
    response = query_index(index, query)
    if len(str(response).strip()) < 20:
        return "[ATTENTION] Donnees insuffisantes dans la base."
    return str(response)


@tool
def get_player_stats(player_name: str) -> str:
    """Recupere les stats d'un joueur specifique depuis la base.
    Args: player_name (ex: "Nordin Amrabat"). Retourne stats ou message si non reference."""
    index = get_index()
    results = retrieve_context(index, f"statistiques joueur {player_name}", top_k=3)
    filtered = _filter_results(results)
    if not filtered:
        return f"[ATTENTION] Joueur '{player_name}' non trouve ou stats insuffisantes."
    out = f"Stats {player_name}:\n"
    for r in filtered:
        out += f"\nSource: {r.get('source','inconnu')}\n{r['text']}\n"
    return out
