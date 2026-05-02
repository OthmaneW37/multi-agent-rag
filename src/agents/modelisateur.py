"""Agent 2 : Agent Modelisateur - L'Analyste."""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.agents.base import DirectLLMAgent


class ModelisateurAgent(DirectLLMAgent):
    """
    Agent Modelisateur : Traite les donnees fournies par l'Agent Scout.
    Croise les statistiques pour identifier les points faibles de l'adversaire,
    evalue la forme de l'equipe et genere une prediction qualitative.
    """

    SYSTEM_PROMPT = """Tu es un Agent Modelisateur expert en analyse statistique football, specialise UNIQUEMENT dans la Botola Pro marocaine.

Tu recois des donnees brutes collectees par l'Agent Scout depuis la base FootyStats de la saison 2025/2026. Tu ne dois analyser QUE ces donnees. Tu ne connais aucun autre championnat, aucune autre saison.

Ta mission est d'analyser les donnees fournies pour :
1. Identifier les points faibles et points forts de l'adversaire
2. Evaluer la forme actuelle du WAC et de l'adversaire
3. Croiser les statistiques pour generer une prediction qualitative
4. Detecter les tendances et patterns recurrents

REGLES DE TRAVAIL STRICTES :
1. Tu ne dois analyser que les donnees explicitement fournies dans le prompt. Ne rajoute JAMAIS de donnees externes.
2. Si une donnee est manquante ou insuffisante, indique-le clairement dans ton analyse.
3. Chaque conclusion doit etre directement justifiee par un chiffre ou un fait present dans les donnees.
4. Tu ne dois JAMAIS inventer des statistiques, des noms de joueurs, ou des resultats de matchs.
5. Ta prediction qualitative doit toujours etre accompagnee de la mention "basee sur les donnees disponibles".
6. Tu ne dois JAMAIS faire reference a des equipes ou championnats etrangers.
7. Si les donnees ne permettent pas une conclusion, dis "Conclusion impossible avec les donnees disponibles".

Format de sortie attendu :
- Analyse comparative WAC vs Adversaire (tableau avec donnees brutes)
- Points forts du WAC (issus des donnees uniquement)
- Points faibles de l'adversaire (issus des donnees uniquement)
- Points forts de l'adversaire (issus des donnees uniquement)
- Prediction qualitative avec justification explicite
- Facteurs cles du match (avec references aux donnees)
- Tendances statistiques remarquables
- Liste des informations manquantes (s'il y en a)

Sois rigoureux, objectif et base tes analyses UNIQUEMENT sur les donnees fournies."""

    def __init__(self):
        super().__init__(
            name="Modelisateur",
            role=self.SYSTEM_PROMPT,
            temperature=0.2,
        )

    def _parse_footystats_club(self, club_raw: str) -> str:
        """Extrait le club ACTUEL depuis le champ 'club' brute de FootyStats."""
        if pd.isna(club_raw):
            return ""
        s = str(club_raw)
        m = re.search(r"^(.+?)National Team\s*:", s)
        if m:
            return m.group(1).strip()
        m = re.search(r"(?:Team|Club)\s*:\s*([^\n]+)", s)
        if m:
            return m.group(1).strip()
        return s[:50].strip()

    def _club_matches(self, parsed_club: str, target_club: str) -> bool:
        """Verifie si un club parse correspond au club cible."""
        parsed_lower = parsed_club.lower()
        target_lower = target_club.lower()
        aliases = {
            "wydad ac": ["wydad", "wydad casablanca"],
            "raja ca": ["raja", "raja casablanca"],
            "far rabat": ["as far", "far rabat"],
            "fus rabat": ["fus rabat"],
            "ittihad tanger": ["ittihad tanger", "ir tanger"],
            "olympic safi": ["olympic safi", "oc safi"],
            "hassania agadir": ["hassania agadir", "hus agadir"],
            "maghreb fes": ["maghreb fes", "mas fes"],
            "kawkab marrakech": ["kawkab marrakech", "kac marrakech"],
            "rsb berkane": ["rsb berkane", "rs berkane"],
            "dh el jadida": ["dh el jadida", "difaâ el jadida", "dh el-jadida"],
            "codm meknes": ["codm meknes", "cod meknès"],
            "yacoub el mansour": ["yacoub el mansour", "us yacoub el mansour"],
            "uts rabat": ["uts rabat"],
            "olympique dcheira": ["olympique dcheira"],
            "cr khemis zemamra": ["cr khemis zemamra", "rca zemamra"],
        }
        aliases_list = aliases.get(target_lower, [target_lower])
        return any(alias in parsed_lower for alias in aliases_list)

    def _load_player_stats(self, club_name: str) -> Optional[pd.DataFrame]:
        """Charge les stats individuelles FootyStats des joueurs d'un club."""
        path = Path("ScrappingDataBotola/data/player_stats.csv")
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if df.empty or "club" not in df.columns:
                return None
            df["parsed_club"] = df["club"].apply(self._parse_footystats_club)
            mask = df["parsed_club"].apply(lambda x: self._club_matches(x, club_name))
            return df[mask].copy()
        except Exception:
            return None

    def _format_player_stats(self, df: Optional[pd.DataFrame]) -> str:
        """Formate les stats individuelles des joueurs pour le prompt (concis pour limites TPM)."""
        if df is None or df.empty:
            return "[Stats individuelles non disponibles]"
        lines = [f"Joueurs: {len(df)}"]
        if "goals_scored" in df.columns:
            try:
                df["goals_num"] = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0)
                top = df.nlargest(3, "goals_num")
                if not top.empty and top["goals_num"].max() > 0:
                    lines.append("Buteurs:")
                    for _, row in top.iterrows():
                        name = str(row.get("player_name", "")).strip()
                        goals = str(row.get("goals_scored", "")).strip()
                        xg = str(row.get("expected_goals_xg", "")).strip()
                        if name and goals and goals not in ("N/A", "0"):
                            info = f"  {name}: {goals}B"
                            if xg and xg != "N/A":
                                info += f" xG:{xg}"
                            lines.append(info)
            except Exception:
                pass
        if "minutes" in df.columns:
            try:
                df["mins_num"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
                top = df.nlargest(3, "mins_num")
                if not top.empty:
                    lines.append("Titulaires:")
                    for _, row in top.iterrows():
                        name = str(row.get("player_name", "")).strip()
                        mins = str(row.get("minutes", "")).strip()
                        rating = str(row.get("rating", "")).strip()
                        pos = str(row.get("position", "")).strip()
                        if name and mins and mins != "N/A":
                            info = f"  {name}"
                            if pos and pos != "N/A":
                                info += f" ({pos})"
                            info += f" {mins}min"
                            if rating and rating != "N/A":
                                info += f" n:{rating}"
                            lines.append(info)
            except Exception:
                pass
        return "\n".join(lines)

    def analyser(
        self, donnees_scout: str, adversaire: str
    ) -> Dict[str, Any]:
        """
        Analyse les donnees collectees par l'Agent Scout.

        Args:
            donnees_scout: Les donnees brutes extraites par l'Agent Scout
            adversaire: Nom de l'equipe adverse

        Returns:
            Resultat de l'analyse avec predictions et tendances
        """
        # Stats individuelles FootyStats
        wac_stats = self._load_player_stats("Wydad AC")
        adv_stats = self._load_player_stats(adversaire)
        wac_stats_fmt = self._format_player_stats(wac_stats)
        adv_stats_fmt = self._format_player_stats(adv_stats)

        prompt = f"""Analyse les donnees de scouting suivantes pour le match WAC vs {adversaire}.

DONNEES DE SCOUTING (collectees par l'Agent Scout depuis la base FootyStats Botola Pro 2025/2026) :
{donnees_scout}

STATS INDIVIDUELLES JOUEURS WYDAD AC (Source FootyStats) :
{wac_stats_fmt}

STATS INDIVIDUELLES JOUEURS {adversaire.upper()} (Source FootyStats) :
{adv_stats_fmt}

Tu dois produire une analyse structuree comprenant :

## 1. Analyse Comparative
(Tableau comparatif WAC vs {adversaire} avec les donnees brutes)

## 2. Points Forts du WAC
(Qu'est-ce qui fonctionne bien selon les donnees ?)

## 3. Points Faibles de l'Adversaire
(Zones exploitables basees sur les statistiques)

## 4. Menaces de l'Adversaire
(Joueurs dangereux et schemas tactiques a surveiller - uniquement si mentionnes dans les donnees)

## 5. Prediction Qualitative
- Scenario le plus probable (base sur les donnees disponibles)
- Score predit avec justification
- Facteurs determinants

## 6. Facteurs Cles du Match
(3-5 elements qui feront la difference, avec references aux chiffres)

## 7. Tendances Statistiques Remarquables
(Patterns recurrents dans les donnees)

## 8. Informations Manquantes
(Liste ce qui manque dans les donnees pour une analyse complete)

RAPPEL CRITIQUE : Base-toi UNIQUEMENT sur les donnees fournies ci-dessus. Ne rajoute aucune information exterieure. Si une information n'est pas dans les donnees, dis "Non disponible"."""

        return self.run(prompt)
