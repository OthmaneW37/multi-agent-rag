"""Agent Rédacteur - Génération de synthèses structurées."""

from typing import Any, Dict

from src.agents.base import BaseAgent
from src.agents.tools import search_documents


class RedacteurAgent(BaseAgent):
    """
    Agent Rédacteur : génère une synthèse structurée avec citations et références
    à partir des analyses et passages collectés.
    """

    SYSTEM_PROMPT = """Tu es un Agent Rédacteur spécialisé dans la rédaction d'états de l'art scientifiques.

Ta mission est de produire une synthèse structurée, claire et académiquement rigoureuse
à partir d'analyses et de passages collectés sur une thématique de recherche.

Règles de travail :
1. Structure la synthèse de manière logique (introduction, corps, conclusion)
2. Intègre systématiquement les citations et références
3. Utilise un style académique formel mais accessible
4. Assure la cohérence entre les différentes sections
5. Inclus une section de discussion et de perspectives
6. Vérifie la fluidité des transitions entre les paragraphes

Format de sortie attendu (état de l'art structuré) :
- Titre
- Résumé/Abstract
- Introduction (contexte, problématique, objectifs)
- Méthodologie de revue
- Synthèse par thématiques
- Discussion (comparaison, tendances, lacunes)
- Conclusion et perspectives
- Références bibliographiques

Sois précis, clair et respectueux des conventions académiques."""

    def __init__(self):
        super().__init__(
            name="Rédacteur",
            role=self.SYSTEM_PROMPT,
            tools=[search_documents],
            temperature=0.5,
        )

    def rediger(self, analyse: str, passages_collectes: str, sujet_recherche: str) -> Dict[str, Any]:
        """
        Rédige une synthèse structurée à partir des analyses et passages.
        
        Args:
            analyse: L'analyse produite par l'agent analyste
            passages_collectes: Les passages extraits par l'agent collecteur
            sujet_recherche: Le sujet de recherche principal
        
        Returns:
            Résultat de la rédaction avec la synthèse complète
        """
        prompt = f"""Rédige un état de l'art complet sur le sujet : "{sujet_recherche}"

ANALYSE FOURNIE PAR L'AGENT ANALYSTE :
{analyse}

PASSAGES DE RÉFÉRENCE (collectés par l'agent collecteur) :
{passages_collectes}

Tu dois produire un document structuré contenant :

# Titre de l'état de l'art

## Résumé
(150-250 mots résumant l'ensemble)

## 1. Introduction
- Contexte et enjeux
- Problématique
- Objectifs de cette revue

## 2. Méthodologie de revue
- Sources consultées
- Critères de sélection

## 3. Synthèse thématique
(Organise par grandes thématiques identifiées dans l'analyse)

## 4. Discussion
- Comparaison des approches
- Tendances identifiées
- Lacunes de la littérature
- Forces et limites des études analysées

## 5. Conclusion et perspectives
- Synthèse des apports majeurs
- Axes de recherche futurs recommandés

## Références
(Liste des sources citées dans le texte)

N'oublie aucune information importante des analyses fournies. Intègre les citations de manière fluide."""

        return self.run(prompt)
