"""Agent 1 : Agent de Scouting - Le Chercheur RAG.

Approche : ReAct maison + outils LangChain.
Le Scout herite de BaseAgent (ReAct) et utilise les outils RAG definis dans tools.py.
Le raisonnement est supervise : l'agent collecte les donnees via outils,
puis structure le rapport avec le LLM.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.agents.base import BaseAgent
from src.agents.tools import search_match_reports, query_sport_database, get_player_stats
from src.rag.indexing import build_or_load_index
from src.rag.retrieval import retrieve_context, query_index


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


class ScoutAgent(BaseAgent):
    """
    Agent de Scouting : collecte les donnees brutes via RAG (outils LangChain)
    puis structure le rapport avec le LLM en mode ReAct.
    """

    SYSTEM_PROMPT = (
        "Tu es un Scout football specialise dans la Botola Pro marocaine. "
        "Tu recois des EXTRAITS de la base FootyStats 2025/2026. "
        "Ta mission : structurer ces donnees en un rapport de scouting factuel. "
        "Regles : base-toi UNIQUEMENT sur les donnees fournies. "
        "Si une info manque, dis 'Non disponible'. N'invente jamais."
    )

    def __init__(self):
        super().__init__(
            name="Scout",
            role=self.SYSTEM_PROMPT,
            tools=[search_match_reports, query_sport_database, get_player_stats],
            temperature=0.2,
        )
        self._index = None

    def _ensure_index(self):
        """Charge l'index RAG une seule fois."""
        if self._index is None:
            print("[Scout] Chargement de l'index RAG...")
            self._index = build_or_load_index()
            print("[Scout] Index charge.")
        return self._index

    def _search(self, query: str, top_k: int = 2) -> str:
        """Recherche RAG directe."""
        try:
            results = retrieve_context(self._ensure_index(), query, top_k=top_k, min_score=0.15)
        except Exception as exc:
            print(f"[Scout][ERREUR RAG] {exc}")
            return f"[Erreur RAG: {exc}]"
        if not results:
            return f"[Aucun resultat pertinent pour : {query}]"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"--- Source {i} (score: {r['score']:.3f}) ---")
            text = r["text"][:300]
            if len(r["text"]) > 300:
                text += "... [tronque]"
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

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
            df = pd.read_csv(path)
            if df.empty or "club" not in df.columns:
                return None

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
        """Formate un effectif en liste ultra-condensee."""
        if df is None or df.empty:
            return "[Effectif non disponible]"
        lines = []
        for pos in ["GK", "DF", "MF", "FW"]:
            pos_players = df[df["Position"] == pos]
            if pos_players.empty:
                continue
            names = pos_players["Player"].head(2).tolist()
            if len(pos_players) > 2:
                names.append("...")
            lines.append(f"{pos}: {', '.join(names)}")
        return " | ".join(lines)

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
        """Formate les stats individuelles en une ligne ultra-condensee."""
        if df is None or df.empty:
            return "[Stats non disponibles]"
        parts = []
        if "goals_scored" in df.columns:
            try:
                df["g"] = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0)
                top = df.nlargest(2, "g")
                if not top.empty and top["g"].max() > 0:
                    names = ", ".join([f"{r['player_name']}({int(r['g'])}B)" for _, r in top.iterrows()])
                    parts.append(f"Buteurs: {names}")
            except Exception:
                pass
        if "minutes" in df.columns:
            try:
                df["m"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
                top = df.nlargest(2, "m")
                if not top.empty:
                    names = ", ".join([f"{r['player_name']}({int(r['m'])}min)" for _, r in top.iterrows()])
                    parts.append(f"Titulaires: {names}")
            except Exception:
                pass
        return " | ".join(parts) if parts else "[Stats non disponibles]"

    def scout_match(self, adversaire: str, contexte: str = "") -> Dict[str, Any]:
        """
        Effectue le scouting pour un match WAC vs adversaire.
        Le Scout utilise le raisonnement ReAct pour collecter les donnees
        via les outils LangChain, puis structure le rapport.
        """
        print(f"[Scout] Recherche des donnees pour WAC vs {adversaire}...")

        # Pre-chargement de l'index pour eviter les conflits threading
        self._ensure_index()

        # ETAPE 1 : Collecte RAG directe (appels deterministes)
        print("[Scout] RAG : forme WAC...")
        wac_forme = self._search("forme WAC stats collectives domicile")

        print("[Scout] RAG : forme adversaire...")
        adv_forme = self._search(f"forme {adversaire} stats collectives")

        print("[Scout] RAG : confrontations...")
        confrontations = self._search(f"confrontations WAC {adversaire}")

        print("[Scout] RAG : stats WAC (query)...")
        try:
            wac_stats_db = query_index(self._ensure_index(), "stats collectives Wydad AC Botola Pro 2025/2026")
        except Exception as exc:
            wac_stats_db = f"[Erreur: {exc}]"

        print("[Scout] RAG : stats adversaire (query)...")
        try:
            adv_stats_db = query_index(self._ensure_index(), f"stats collectives {adversaire} Botola Pro 2025/2026")
        except Exception as exc:
            adv_stats_db = f"[Erreur: {exc}]"

        # Recherche RAG pour date et calendrier
        upcoming = self._search("prochain match Wydad AC upcoming")
        fixtures = self._search("fixtures complete calendrier botola")

        # ETAPE 2 : Effectifs et stats individuelles (donnees CSV)
        wac_stats = self._load_player_stats("Wydad AC")
        adv_stats = self._load_player_stats(adversaire)
        wac_squad = self._load_squad_footystats("Wydad AC")
        adv_squad = self._load_squad_footystats(adversaire)

        # Extraction programmatique de la date du prochain match
        upcoming_date = "Non disponible"
        for line in upcoming.splitlines():
            if "vs" in line and any(x in line for x in [adversaire, adversaire.replace("CA", "Casablanca"), adversaire.replace("Casablanca", "CA")]):
                date_match = re.search(r"\[([^\]]+)\]", line)
                if date_match:
                    upcoming_date = date_match.group(1)
                    break

        # ETAPE 3 : Prompt pour le ReAct
        prompt = (
            f"Tu dois produire un rapport de scouting pour le match WAC vs {adversaire}. "
            f"Contexte: {contexte}\n\n"
            f"=== DONNEES COLLECTEES VIA RAG ===\n\n"
            f"-- Forme WAC (source FootyStats) --\n{wac_forme}\n\n"
            f"-- Forme {adversaire} (source FootyStats) --\n{adv_forme}\n\n"
            f"-- Confrontations directes --\n{confrontations}\n\n"
            f"-- Stats WAC (query engine) --\n{wac_stats_db}\n\n"
            f"-- Stats {adversaire} (query engine) --\n{adv_stats_db}\n\n"
            f"-- Calendrier / upcoming --\n{upcoming}\n\n"
            f"-- Fixtures --\n{fixtures}\n\n"
            f"=== EFFECTIFS ET STATS INDIVIDUELLES ===\n"
            f"WAC: {self._format_squad(wac_squad)} | {self._format_player_stats(wac_stats)}\n"
            f"{adversaire}: {self._format_squad(adv_squad)} | {self._format_player_stats(adv_stats)}\n\n"
            f"Produis le rapport de scouting STRICTEMENT avec ce format Markdown:\n\n"
            f"## 1. Contexte du Match\n"
            f"(date: {upcoming_date}, competition, stade)\n\n"
            f"## 2. Forme du WAC\n"
            f"(stats collectives)\n\n"
            f"## 3. Forme de {adversaire}\n"
            f"(stats collectives)\n\n"
            f"## 4. Confrontations Directes\n"
            f"(historique)\n\n"
            f"## 5. Analyse des Tendances\n"
            f"(BTTS, corners, cartons, timing buts)\n\n"
            f"## 6. Joueurs Cles\n"
            f"(noms reels issus des effectifs fournis)\n\n"
            f"## 7. Faits Saillants\n"
            f"(blessures, suspensions)\n\n"
            f"## 8. Informations Manquantes\n"
            f"(liste ce qui manque)\n\n"
            f"Base-toi UNIQUEMENT sur les donnees ci-dessus. Ne rajoute RIEN d'exterieur."
        )

        result = self.run(prompt)

        # Fallback reformatage si le LLM n'a pas respecte le format Markdown
        raw_output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
        if "##" not in raw_output:
            print("[Scout] Format Markdown manquant. Reformattage force...")
            reformat_prompt = (
                f"Reformate le texte suivant en un rapport de scouting Markdown "
                f"avec EXACTEMENT ces sections :\n\n"
                f"## 1. Contexte du Match\n"
                f"(date, competition, stade)\n\n"
                f"## 2. Forme du WAC\n"
                f"(stats collectives)\n\n"
                f"## 3. Forme de {adversaire}\n"
                f"(stats collectives)\n\n"
                f"## 4. Confrontations Directes\n"
                f"(historique)\n\n"
                f"## 5. Analyse des Tendances\n"
                f"(BTTS, corners, cartons, timing buts)\n\n"
                f"## 6. Joueurs Cles\n"
                f"(noms reels issus des effectifs fournis)\n\n"
                f"## 7. Faits Saillants\n"
                f"(blessures, suspensions)\n\n"
                f"## 8. Informations Manquantes\n"
                f"(liste ce qui manque)\n\n"
                f"Texte a reformater :\n\n{raw_output}\n\n"
                f"Regles : base-toi UNIQUEMENT sur le texte fourni. "
                f"Si une info manque pour une section, ecris 'Non disponible'."
            )
            reformat_result = self.run(reformat_prompt)
            reformat_output = (
                reformat_result.get("output", str(reformat_result))
                if isinstance(reformat_result, dict)
                else str(reformat_result)
            )
            if "##" in reformat_output:
                raw_output = reformat_output
                if isinstance(result, dict):
                    result["output"] = raw_output
                else:
                    result = {"output": raw_output}
            else:
                # Dernier recours : encapsuler le texte brut dans une structure minimale
                raw_output = (
                    f"## 1. Contexte du Match\n\n"
                    f"Date et heure: {upcoming_date}\n\n"
                    f"## 2. Forme du WAC\n\n"
                    f"Non disponible\n\n"
                    f"## 3. Forme de {adversaire}\n\n"
                    f"{raw_output}\n\n"
                    f"## 4. Confrontations Directes\n\n"
                    f"Non disponible\n\n"
                    f"## 5. Analyse des Tendances\n\n"
                    f"Non disponible\n\n"
                    f"## 6. Joueurs Cles\n\n"
                    f"Non disponible\n\n"
                    f"## 7. Faits Saillants\n\n"
                    f"Non disponible\n\n"
                    f"## 8. Informations Manquantes\n\n"
                    f"Donnees RAG insuffisantes pour ce match."
                )
                if isinstance(result, dict):
                    result["output"] = raw_output
                else:
                    result = {"output": raw_output}

        # Post-processing : injection de la date
        if upcoming_date != "Non disponible" and "output" in result:
            output = result["output"]
            output = re.sub(
                r"(Date et heure:\s*)Non disponible",
                rf"\g<1>{upcoming_date}",
                output,
                flags=re.IGNORECASE,
            )
            result["output"] = output

        return result
