"""Agent Collecteur - Recherche et extraction de passages pertinents."""

from typing import Any, Dict

from src.agents.base import BaseAgent
from src.agents.tools import search_documents, query_knowledge_base


class CollecteurAgent(BaseAgent):
    """
    Agent Collecteur : parcourt le corpus indexé via RAG et extrait les passages pertinents
    pour une thématique de recherche donnée.
    """

    SYSTEM_PROMPT = """Tu es un Agent Collecteur spécialisé dans la recherche documentaire académique.

Ta mission est de parcourir un corpus de publications scientifiques indexées via RAG et d'extraire 
les passages les plus pertinents pour une thématique de recherche donnée.

Règles de travail :
1. Utilise l'outil 'search_documents' pour trouver les documents pertinents
2. Utilise l'outil 'query_knowledge_base' pour obtenir des réponses synthétisées
3. Pour chaque requête, effectue plusieurs recherches avec des angles différents
4. Organise les résultats par thème et par pertinence
5. Note toujours la source (titre/article) de chaque passage extrait
6. Privilégie la diversité des sources

Format de sortie attendu :
- Liste des thèmes identifiés
- Pour chaque thème : passages pertinents avec citations
- Sources complètes (auteur, titre, année si disponible)
- Score de pertinence estimé pour chaque passage

Sois exhaustif et méthodique dans ta collecte."""

    def __init__(self):
        super().__init__(
            name="Collecteur",
            role=self.SYSTEM_PROMPT,
            tools=[search_documents, query_knowledge_base],
            temperature=0.3,
        )

    def collecter(self, sujet_recherche: str) -> Dict[str, Any]:
        """
        Collecte les passages pertinents pour un sujet de recherche.
        
        Args:
            sujet_recherche: Le sujet ou la question de recherche
        
        Returns:
            Résultat de la collecte avec passages et sources
        """
        prompt = f"""Effectue une collecte exhaustive sur le sujet suivant : "{sujet_recherche}"

Tu dois :
1. Rechercher les documents pertinents avec plusieurs angles de recherche
2. Extraire les passages clés avec leurs sources
3. Organiser les résultats par thèmes
4. Retourner une collection structurée de passages pertinents

Commence par une recherche générale, puis affine avec des requêtes plus spécifiques."""

        return self.run(prompt)
