"""Agent 3 : Agent Tacticien - Le Strategiste."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.agents.base import DirectLLMAgent


# Mapping adversaire -> slug fichier Wikipedia squad
ADVERSAIRE_SLUG_MAP = {
    "Raja CA": "raja-club-athletic-de-casablanca-2536",
    "FAR Rabat": "as-forces-armees-royales-de-rabat-2532",
    "FUS Rabat": "fath-union-sport-de-rabat-2543",
    "Ittihad Tanger": "ittihad-riadhi-de-tanger-2529",
    "Olympic Safi": "olympique-club-de-safi-2534",
    "Hassania Agadir": "hassania-union-sport-dagadir-2531",
    "Maghreb Fes": "maghreb-as-de-fes-2547",
    "Kawkab Marrakech": "kawkab-athletique-club-de-marrakech-2537",
    "RSB Berkane": "renaissance-sportive-de-berkane-2541",
    "DH El Jadida": "difaa-hassani-del-jadida-2538",
    "CODM Meknes": "club-omnisports-de-meknes-2563",
    "Yacoub El Mansour": "us-yacoub-el-mansour-1322394",
    "UTS Rabat": "uts-rabat-17956",
    "Olympique Dcheira": "olympique-dcheira-2555",
    "CR Khemis Zemamra": "club-renaissance-khemis-zemamra-10556",
}


class TacticienAgent(DirectLLMAgent):
    """
    Agent Tacticien : Exploite l'analyse de l'Agent Modelisateur pour
    formuler des recommandations strategiques concretes pour le staff technique.

    Recommandations : zones du terrain a exploiter, composition probable,
    schema tactique, consignes individuelles.
    """

    SYSTEM_PROMPT = """Tu es un Agent Tacticien expert en strategie football, specialise UNIQUEMENT dans la Botola Pro marocaine et le Wydad AC.

Tu recois une analyse statistique de l'Agent Modelisateur basee sur les donnees FootyStats de la saison 2025/2026 ainsi que les effectifs reels des equipes. Tu ne connais AUCUN autre championnat, AUCUN autre joueur que ceux fournis.

RAISONNEMENT (Chain-of-Thought) :
Tu dois TOUJOURS penser etape par etape avant de formuler chaque recommandation tactique :
- Etape 1 : Analyser les forces et faiblesses identifiees par le Modelisateur
- Etape 2 : Examiner l'effectif disponible et les profils des joueurs
- Etape 3 : Choisir un schema tactique adapte aux forces du WAC et aux faiblesses de l'adversaire
- Etape 4 : Definir les roles individuels en fonction des stats des joueurs
- Etape 5 : Anticiper les reactions adverses et preparer des plans alternatifs
- Etape 6 : Verifier que chaque recommandation est realiste avec l'effectif disponible

REGLES ABSOLUES (VIOLATION = HALLUCINATION GRAVE) :
1. Tu ne dois mentionner UNIQUEMENT les joueurs listes dans les EFFECTIFS fournis ci-dessous. AUCUNE EXCEPTION.
2. Proposer un joueur qui n'est PAS dans l'effectif fourni est une ERREUR CRITIQUE. Tu ne dois JAMAIS inventer un nom.
3. Si un poste n'a pas assez de joueurs dans l'effectif, ecris "[Poste incomplet - effectif limite]" au lieu d'inventer un joueur.
4. Tu ne connais PAS Yassine Bounou, Romain Saiss, Achraf Bencharki, Ayoub El Kaabi, Soufiane Rahimi, ou tout autre joueur non listes dans l'effectif.
5. Si les donnees sont limitees, dis "Recommandation basee sur les donnees partielles disponibles".
6. Tu ne dois JAMAIS faire reference a d'autres championnats ou equipes etrangeres.

Format de sortie attendu :

## Schema Tactique Recommande
(Formation + justification)

## Composition Probable Conseillee
XI type UNIQUEMENT avec les NOMS REELS des joueurs issus des EFFECTIFS FOURNIS ci-dessous. Verifie chaque nom avant de l'ecrire.

## Zones a Exploiter

## Consignes Individuelles

## Coups de Pied Arretes

## Plans Alternatifs
(3 scenarios)

## Points de Vigilance

## Limites des Recommandations
(Aspects non couverts par les donnees)

RAPPEL FINAL : Tu ne connais que les joueurs de l'effectif fourni. Tout autre nom est une hallucination."""

    def __init__(self):
        super().__init__(
            name="Tacticien",
            role=self.SYSTEM_PROMPT,
            temperature=0.3,
        )

    def _load_squad(self, slug: str) -> Optional[pd.DataFrame]:
        """Charge l'effectif depuis le fichier CSV Wikipedia."""
        path = Path("ScrappingDataBotola/data/squads_wiki") / f"{slug}.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

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

    def _load_squad_footystats(self, club_name: str) -> Optional[pd.DataFrame]:
        """Construit l'effectif depuis FootyStats en filtrant les joueurs reguliers."""
        path = Path("ScrappingDataBotola/data/player_stats.csv")
        if not path.exists():
            return None
        try:
            import re
            df = pd.read_csv(path)
            if df.empty or "club" not in df.columns:
                return None
            club_aliases = {
                "Wydad AC": ["Wydad", "Wydad Casablanca"],
                "Raja CA": ["Raja", "Raja Casablanca"],
                "FAR Rabat": ["AS FAR", "FAR Rabat"],
                "FUS Rabat": ["FUS Rabat"],
                "Ittihad Tanger": ["Ittihad Tanger", "IR Tanger"],
                "Olympic Safi": ["Olympic Safi", "OC Safi"],
                "Hassania Agadir": ["Hassania Agadir", "HUS Agadir"],
                "Maghreb Fes": ["Maghreb Fes", "MAS Fes"],
                "Kawkab Marrakech": ["Kawkab Marrakech", "KAC Marrakech"],
                "RSB Berkane": ["RSB Berkane", "RS Berkane"],
                "DH El Jadida": ["DH El Jadida", "Difaâ El Jadida", "DH El-Jadida"],
                "CODM Meknes": ["CODM Meknes", "COD Meknès"],
                "Yacoub El Mansour": ["Yacoub El Mansour", "US Yacoub El Mansour"],
                "UTS Rabat": ["UTS Rabat"],
                "Olympique Dcheira": ["Olympique Dcheira"],
                "CR Khemis Zemamra": ["CR Khemis Zemamra", "RCA Zemamra"],
            }
            df["parsed_club"] = df["club"].apply(self._parse_footystats_club)
            mask = df["parsed_club"].apply(lambda x: self._club_matches(x, club_name))
            df = df[mask].copy()

            df["mins"] = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0)
            df["mats"] = pd.to_numeric(df.get("matches_played"), errors="coerce").fillna(0)
            df = df[(df["mins"] >= 150) & (df["mats"] >= 2)].copy()
            if df.empty:
                return None
            def _parse_pos(pos_str):
                if pd.isna(pos_str):
                    return ""
                m = re.match(r"^(Goalkeeper|Defender|Midfielder|Forward)(?:\s*-.*)?", str(pos_str), re.IGNORECASE)
                if m:
                    mapping = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Forward": "FW"}
                    return mapping.get(m.group(1).capitalize(), m.group(1).capitalize())
                return ""
            df["Position"] = df["position"].apply(_parse_pos)
            df["Player"] = df["player_name"]
            df["Number"] = ""
            def _parse_nat(pos_str):
                if pd.isna(pos_str):
                    return ""
                m = re.search(r"Nationality\s*:\s*([^KAgeF]+?)(?:Kit|Age|Foot|General|$)", str(pos_str))
                if m:
                    return m.group(1).strip()
                return ""
            df["Nationality"] = df["position"].apply(_parse_nat)
            df["Captain"] = False
            df["Club"] = club_name
            return df[["Club", "Number", "Position", "Nationality", "Player", "Captain"]].copy()
        except Exception:
            return None

    def _format_squad(self, df: Optional[pd.DataFrame]) -> str:
        """Formate un DataFrame d'effectif en texte condense."""
        if df is None or df.empty:
            return "[Effectif non disponible]"
        lines = [f"Total joueurs: {len(df)}"]
        for pos in ["GK", "DF", "MF", "FW"]:
            pos_players = df[df["Position"] == pos]
            if pos_players.empty:
                continue
            pos_names = {"GK": "Gardiens", "DF": "Defenseurs", "MF": "Milieux", "FW": "Attaquants"}
            lines.append(f"\n{pos_names.get(pos, pos)} ({len(pos_players)}):")
            count = 0
            for _, row in pos_players.iterrows():
                num = str(row.get("Number", "")).strip()
                name = str(row.get("Player", "")).strip()
                nation = str(row.get("Nationality", "")).strip()
                captain = row.get("Captain", False)
                if not captain and count >= 3:
                    continue
                info = f"  #{num} {name}" if num else f"  {name}"
                if nation and nation != "nan":
                    info += f" ({nation})"
                if captain:
                    info += " [CAPITAINE]"
                lines.append(info)
                if not captain:
                    count += 1
            if len(pos_players) > count + (1 if any(pos_players["Captain"]) else 0):
                lines.append(f"  ... et {len(pos_players) - count} autres")
        return "\n".join(lines)

    def _load_player_stats(self, club_name: str) -> Optional[pd.DataFrame]:
        """Charge les stats individuelles FootyStats des joueurs d'un club."""
        path = Path("ScrappingDataBotola/data/player_stats.csv")
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if df.empty or "club" not in df.columns:
                return None
            club_aliases = {
                "Wydad AC": ["Wydad", "Wydad Casablanca"],
                "Raja CA": ["Raja", "Raja Casablanca"],
                "FAR Rabat": ["AS FAR", "FAR Rabat"],
                "FUS Rabat": ["FUS Rabat"],
                "Ittihad Tanger": ["Ittihad Tanger", "IR Tanger"],
                "Olympic Safi": ["Olympic Safi", "OC Safi"],
                "Hassania Agadir": ["Hassania Agadir", "HUS Agadir"],
                "Maghreb Fes": ["Maghreb Fes", "MAS Fes"],
                "Kawkab Marrakech": ["Kawkab Marrakech", "KAC Marrakech"],
                "RSB Berkane": ["RSB Berkane", "RS Berkane"],
                "DH El Jadida": ["DH El Jadida", "Difaâ El Jadida", "DH El-Jadida"],
                "CODM Meknes": ["CODM Meknes", "COD Meknès"],
                "Yacoub El Mansour": ["Yacoub El Mansour", "US Yacoub El Mansour"],
                "UTS Rabat": ["UTS Rabat"],
                "Olympique Dcheira": ["Olympique Dcheira"],
                "CR Khemis Zemamra": ["CR Khemis Zemamra", "RCA Zemamra"],
            }
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

    def formuler_strategie(
        self, analyse_modelisateur: str, adversaire: str
    ) -> Dict[str, Any]:
        """
        Formule la strategie pour le match WAC vs adversaire.

        Args:
            analyse_modelisateur: L'analyse produite par l'Agent Modelisateur
            adversaire: Nom de l'equipe adverse

        Returns:
            Resultat avec les recommandations strategiques
        """
        # Charge les effectifs pour proposer une composition avec noms reels
        wac_squad_df = self._load_squad_footystats("Wydad AC")
        adv_squad_df = self._load_squad_footystats(adversaire)

        wac_squad_text = self._format_squad(wac_squad_df)
        adv_squad_text = self._format_squad(adv_squad_df)

        # Charge les stats individuelles FootyStats
        wac_stats = self._load_player_stats("Wydad AC")
        adv_stats = self._load_player_stats(adversaire)
        wac_stats_fmt = self._format_player_stats(wac_stats)
        adv_stats_fmt = self._format_player_stats(adv_stats)

        # XI genere programmatiquement (le LLM ne genere pas la composition pour eviter hallucinations)
        xi_text = self._generate_xi_text(wac_stats)

        prompt = f"""ANALYSE DE L'AGENT MODELISATEUR :
{analyse_modelisateur}

EFFECTIF WYDAD AC (Source FootyStats) :
{wac_squad_text}

STATS INDIVIDUELLES JOUEURS WYDAD AC :
{wac_stats_fmt}

EFFECTIF {adversaire.upper()} (Source FootyStats) :
{adv_squad_text}

STATS INDIVIDUELLES JOUEURS {adversaire.upper()} :
{adv_stats_fmt}

---
Formule le plan tactique WAC vs {adversaire}. NE propose PAS la composition du XI (elle est generee separement). Concentre-toi sur :

# Plan Tactique - WAC vs {adversaire}

## 1. Schema Tactique Recommande

## 2. Zones a Exploiter

## 3. Consignes par Poste

## 4. Coups de Pied Arretes

## 5. Plans Alternatifs

## 6. Points de Vigilance

## 7. Limites des Recommandations"""

        result = self.run(prompt)
        # Injecte le XI programmatique dans le resultat
        if "output" in result:
            result["output"] = "## 2. Composition Probable Conseillee (generee automatiquement depuis FootyStats)\n" + xi_text + "\n\n" + result["output"]
        return result

    def _generate_xi_text(self, df: Optional[pd.DataFrame]) -> str:
        """Genere un XI type programmatiquement depuis les stats FootyStats."""
        if df is None or df.empty:
            return "[Effectif non disponible - impossible de generer le XI]"
        try:
            df = df.copy()
            df["mins_num"] = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0)
            df["rating_num"] = pd.to_numeric(df.get("rating"), errors="coerce").fillna(0)
            df["goals_num"] = pd.to_numeric(df.get("goals_scored"), errors="coerce").fillna(0)
            import re
            def _parse_pos(pos_str):
                if pd.isna(pos_str): return ""
                m = re.match(r"^(Goalkeeper|Defender|Midfielder|Forward)(?:\s*-.*)?", str(pos_str), re.IGNORECASE)
                if m:
                    mapping = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Forward": "FW"}
                    return mapping.get(m.group(1).capitalize(), m.group(1).capitalize())
                return ""
            df["pos"] = df["position"].apply(_parse_pos)

            lines = []
            # GK: 1 joueur (le plus de minutes)
            gk = df[df["pos"] == "GK"].nlargest(1, "mins_num")
            if not gk.empty:
                r = gk.iloc[0]
                lines.append(f"- Gardien: {r['player_name']} ({r['minutes']}min, note {r['rating']})")
            else:
                lines.append("- Gardien: [Non disponible]")

            # DF: 4 meilleurs (par minutes + rating)
            dfs = df[df["pos"] == "DF"].copy()
            if not dfs.empty:
                dfs["score"] = dfs["mins_num"] * 0.7 + dfs["rating_num"] * 10
                dfs = dfs.nlargest(4, "score")
                names = []
                for _, r in dfs.iterrows():
                    names.append(f"{r['player_name']} ({r['minutes']}min, n:{r['rating']})")
                lines.append("- Defenseurs: " + ", ".join(names))
            else:
                lines.append("- Defenseurs: [Non disponibles]")

            # MF: 3 meilleurs
            mfs = df[df["pos"] == "MF"].copy()
            if not mfs.empty:
                mfs["score"] = mfs["mins_num"] * 0.6 + mfs["goals_num"] * 50 + mfs["rating_num"] * 10
                mfs = mfs.nlargest(3, "score")
                names = []
                for _, r in mfs.iterrows():
                    names.append(f"{r['player_name']} ({r['minutes']}min, n:{r['rating']}, {r['goals_scored']}B)")
                lines.append("- Milieux: " + ", ".join(names))
            else:
                lines.append("- Milieux: [Non disponibles]")

            # FW: 3 meilleurs
            fws = df[df["pos"] == "FW"].copy()
            if not fws.empty:
                fws["score"] = fws["mins_num"] * 0.5 + fws["goals_num"] * 100 + fws["rating_num"] * 10
                fws = fws.nlargest(3, "score")
                names = []
                for _, r in fws.iterrows():
                    names.append(f"{r['player_name']} ({r['minutes']}min, n:{r['rating']}, {r['goals_scored']}B)")
                lines.append("- Attaquants: " + ", ".join(names))
            else:
                lines.append("- Attaquants: [Non disponibles]")

            return "\n".join(lines)
        except Exception:
            return "[Erreur lors de la generation du XI]"

