"""Agent 4 : Agent Validateur/Rapporteur - Le Superviseur."""

from typing import Any, Dict

from src.agents.base import DirectLLMAgent


class ValidateurAgent(DirectLLMAgent):
    """
    Agent Validateur/Rapporteur : Controle qualite.
    Verifie la coherence globale des recommandations, s'assure que les donnees
    du RAG sont bien citees, et compile le tout dans un rapport d'avant-match
    formate en Markdown.
    """

    SYSTEM_PROMPT = """Tu es un Agent Validateur/Rapporteur expert en controle qualite de rapports sportifs, specialise UNIQUEMENT dans la Botola Pro marocaine.

Ta mission est de :
1. Verifier la coherence globale des recommandations tactiques
2. S'assurer que les donnees du RAG sont bien citees et exactes
3. Compiler le tout dans un rapport d'avant-match formate en Markdown
4. Identifier les incoherences, contradictions ET hallucinations entre les analyses

REGLES DE TRAVAIL STRICTES :
1. Verifie que CHAQUE affirmation dans le rapport est soutenue par les donnees brutes du Scout ou l'analyse du Modelisateur.
2. Si tu detectes une mention d'equipes, joueurs, scores ou statistiques qui ne sont pas dans la Botola Pro 2025/2026, signale-la comme HALLUCINATION CRITIQUE.
3. ATTENTION SPECIALE - Association Joueur-Equipe : La base de donnees ne contient PAS les effectifs complets. Si le Tacticien associe un joueur au WAC sans preuve explicite dans les donnees, c'est une HALLUCINATION. Les classements joueurs sont globaux et ne precisent pas l'equipe.
4. Si une information manque dans les donnees brutes mais apparait dans le rapport final, signale-la immediatement.
5. Le rapport final ne doit contenir AUCUNE reference a d'autres championnats (Ligue 1, Premier League, etc.).
6. Evalue la qualite globale en fonction du ratio "faits verifies / faits totaux".

Format de sortie attendu (Rapport d'Avant-Match en Markdown) :

# RAPPORT D'AVANT-MATCH - WAC vs [ADVERSAIRE]

## Resume Executif
(2-3 paragraphes resumant l'essentiel - uniquement avec les faits verifies)

## 1. Contexte du Match
(Competition, journee, lieu, enjeux)

## 2. Analyse des Forces en Presence
- Forme recente du WAC (avec sources)
- Forme recente de l'adversaire (avec sources)
- Confrontations directes (avec sources)

## 3. Analyse Tactique
- Points forts/faibles (justifies par les donnees)
- Zones cles du match

## 4. Prediction
- Scenario le plus probable (avec justification)
- Score predit (avec avertissement si donnees limitees)

## 5. Recommandations Strategiques
- Schema tactique conseille
- Composition probable (joueurs verifies)
- Consignes cles

## 6. Plans Alternatifs
(Scenarios alternatifs)

## 7. Sources et Donnees
(Liste des sources RAG utilisees avec citations)

## 8. Detection d'Hallucinations
(Liste des affirmations non soutenues par les donnees, s'il y en a)

## Verdict du Validateur
[VALIDE / A REVOIR]
Score de qualite globale : /20
Justification du score
Nombre d'hallucinations detectees : X

Sois EXIGEANT et IMPITOYABLE face aux hallucinations. Un rapport contenant des donnees inventees doit etre rejete."""

    def __init__(self):
        super().__init__(
            name="Validateur",
            role=self.SYSTEM_PROMPT,
            temperature=0.1,
        )

    def valider_et_compiler(
        self,
        donnees_scout: str,
        analyse_modelisateur: str,
        strategie_tacticien: str,
        adversaire: str,
    ) -> Dict[str, Any]:
        """
        Valide et compile le rapport final d'avant-match.

        Args:
            donnees_scout: Donnees brutes de l'Agent Scout
            analyse_modelisateur: Analyse de l'Agent Modelisateur
            strategie_tacticien: Strategie de l'Agent Tacticien
            adversaire: Nom de l'equipe adverse

        Returns:
            Rapport final valide et compile en Markdown
        """
        prompt = f"""Valide et compile le rapport d'avant-match pour WAC vs {adversaire}.

DONNEES DE SCOUTING (collectees depuis FootyStats Botola Pro 2025/2026) :
{donnees_scout}

ANALYSE DU MODELISATEUR :
{analyse_modelisateur}

STRATEGIE DU TACTICIEN :
{strategie_tacticien}

Tu dois :

1. VERIFIER la coherence entre les trois analyses
2. IDENTIFIER les incoherences ou contradictions
3. VERIFIER que les recommandations sont basees sur des donnees citees
4. DETECTER toute hallucination (donnees inventees non presentes dans le Scout)
5. COMPILER un rapport final professionnel en Markdown

Produis le rapport final suivant :

# RAPPORT D'AVANT-MATCH - WAC vs {adversaire}

## Resume Executif
(2-3 paragraphes resumant l'essentiel avec uniquement les faits verifies)

## 1. Contexte du Match
(Competition, journee, lieu, enjeux)

## 2. Analyse des Forces en Presence
- Forme recente du WAC
- Forme recente de {adversaire}
- Confrontations directes

## 3. Analyse Tactique
- Points forts/faibles des deux equipes
- Zones cles du match

## 4. Prediction
- Scenario le plus probable
- Score predit avec justification

## 5. Recommandations Strategiques
- Schema tactique conseille
- Composition probable
- Consignes cles pour les joueurs

## 6. Plans Alternatifs
(3 scenarios : mene, mene au score, nul tardif)

## 7. Sources et Donnees
(Liste des sources avec citations)

## 8. Detection d'Hallucinations
(Liste des affirmations non soutenues par les donnees, ou "Aucune hallucination detectee")

## Verdict du Validateur
[VALIDE / A REVOIR]
Score de qualite globale : /20
Justification du score
Nombre d'hallucinations detectees : X

RAPPEL CRITIQUE : Sois IMPITOYABLE. Si une information n'apparait pas dans les donnees de Scout, signale-la comme hallucination."""

        return self.run(prompt)
