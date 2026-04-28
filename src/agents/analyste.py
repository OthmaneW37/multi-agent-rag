"""Agent Analyste - Analyse des tendances et comparaison des approches."""

from typing import Any, Dict

from src.agents.base import BaseAgent
from src.agents.tools import search_documents


class AnalysteAgent(BaseAgent):
    """
    Agent Analyste : identifie les tendances, compare les approches, détecte les lacunes
    dans un ensemble de passages scientifiques.
    """

    SYSTEM_PROMPT = """Tu es un Agent Analyste spécialisé dans l'analyse critique de littérature scientifique.

Ta mission est d'analyser un ensemble de passages extraits de publications scientifiques pour :
1. Identifier les tendances majeures et émergentes
2. Comparer les différentes approches méthodologiques
3. Détecter les lacunes et les contradictions dans la littérature
4. Évaluer la qualité et la robustesse des travaux cités

Règles de travail :
1. Analyse systématiquement les méthodologies employées
2. Compare les résultats et conclusions des différentes études
3. Identifie les axes de recherche sous-explorés
4. Détecte les biais potentiels ou les limites des études
5. Classe les travaux par écoles de pensée ou approches

Format de sortie attendu :
- Synthèse des tendances identifiées
- Tableau comparatif des approches (avantages/inconvénients)
- Lacunes de recherche détectées
- Recommandations pour de futures études
- Évaluation critique de la qualité des sources

Sois rigoureux, objectif et constructif dans ton analyse."""

    def __init__(self):
        super().__init__(
            name="Analyste",
            role=self.SYSTEM_PROMPT,
            tools=[search_documents],
            temperature=0.4,
        )

    def analyser(self, passages_collectes: str, sujet_recherche: str) -> Dict[str, Any]:
        """
        Analyse les passages collectés pour une thématique donnée.
        
        Args:
            passages_collectes: Les passages extraits par l'agent collecteur
            sujet_recherche: Le sujet de recherche principal
        
        Returns:
            Résultat de l'analyse avec tendances, comparaisons et lacunes
        """
        prompt = f"""Analyse les passages scientifiques suivants sur le sujet "{sujet_recherche}".

PASSAGES COLLECTÉS :
{passages_collectes}

Tu dois produire une analyse structurée comprenant :
1. **Tendances majeures** : Quelles sont les orientations dominantes ?
2. **Comparaison des approches** : Tableau comparatif des méthodologies
3. **Lacunes détectées** : Qu'est-ce qui manque dans la littérature ?
4. **Contradictions** : Y a-t-il des résultats opposés ?
5. **Qualité des sources** : Évaluation critique
6. **Recommandations** : Axes prometteurs pour la recherche future

Si nécessaire, effectue des recherches complémentaires pour valider tes analyses."""

        return self.run(prompt)
