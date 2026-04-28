"""Agent Vérificateur - Validation de la qualité et de la cohérence."""

from typing import Any, Dict

from src.agents.base import BaseAgent
from src.agents.tools import search_documents


class VerificateurAgent(BaseAgent):
    """
    Agent Vérificateur : vérifie la cohérence, les citations, et la qualité du rendu final.
    """

    SYSTEM_PROMPT = """Tu es un Agent Vérificateur spécialisé dans le contrôle qualité de documents académiques.

Ta mission est de vérifier rigoureusement la qualité, la cohérence et l'exactitude d'un état de l'art
produit par un autre agent, en te basant sur le corpus de documents scientifiques indexés.

Règles de travail :
1. Vérifie la cohérence logique de l'ensemble du document
2. Contrôle l'exactitude des citations par rapport aux sources originales
3. Évalue la couverture de la littérature (y a-t-il des omissions importantes ?)
4. Vérifie la qualité rédactionnelle (style, orthographe, fluidité)
5. Contrôle la pertinence des conclusions par rapport aux données présentées
6. Identifie d'éventuelles hallucinations ou inventions de l'agent rédacteur

Format de sortie attendu :
- Rapport de validation global (VALIDÉ / À REVOIR / REJETÉ)
- Liste des points forts
- Liste des erreurs/incohérences détectées (avec gravité : mineure/majeure/critique)
- Suggestions de corrections
- Vérification des citations (présentes ? exactes ?)
- Score de qualité global (/20)

Sois exigeant mais constructif. Ton objectif est d'améliorer la qualité finale du document."""

    def __init__(self):
        super().__init__(
            name="Verificateur",
            role=self.SYSTEM_PROMPT,
            tools=[search_documents],
            temperature=0.2,
        )

    def verifier(self, document: str, passages_sources: str, sujet_recherche: str) -> Dict[str, Any]:
        """
        Vérifie la qualité et la cohérence du document final.
        
        Args:
            document: Le document produit par l'agent rédacteur
            passages_sources: Les passages sources originaux pour vérification
            sujet_recherche: Le sujet de recherche principal
        
        Returns:
            Résultat de la vérification avec rapport qualité
        """
        prompt = f"""Vérifie le document suivant sur le sujet "{sujet_recherche}".

DOCUMENT À VÉRIFIER :
{document}

SOURCES ORIGINALES (pour vérification des citations) :
{passages_sources}

Tu dois produire un RAPPORT DE VALIDATION détaillé :

## Verdict Global
[VALIDÉ / À REVOIR / REJETÉ]

## Points Forts
(Liste des aspects réussis)

## Erreurs et Incohérences Détectées
Pour chaque problème :
- **Localisation** : où dans le document ?
- **Description** : quel est le problème ?
- **Gravité** : Mineure / Majeure / Critique
- **Suggestion de correction**

## Vérification des Citations
- Citations présentes et bien formatées ?
- Correspondance avec les sources originales ?
- Omissions importantes ?

## Qualité Rédactionnelle
- Style académique respecté ?
- Fluidité et cohérence ?
- Structure logique ?

## Score de Qualité Global
(/20 avec justification)

## Recommandations Finales
Si des corrections sont nécessaires, détaille-les clairement."""

        return self.run(prompt)
