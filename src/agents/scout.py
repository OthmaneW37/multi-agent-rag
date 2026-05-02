"""Agent 1 : Agent de Scouting - Le Chercheur RAG."""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.agents.base import DirectLLMAgent
from src.rag.indexing import build_or_load_index
from src.rag.retrieval import retrieve_context


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


class ScoutAgent(DirectLLMAgent):
    """
    Agent de Scouting : Parcourt la base vectorielle indexee par LlamaIndex
    pour extraire les donnees brutes et faits saillants concernant le WAC
    et son prochain adversaire.

    Ce agent effectue les recherches RAG directement en code (pas via ReAct)
    pour eviter les multiples appels LLM lents avec Ollama.
    """

    SYSTEM_PROMPT = """Tu es un Agent de Scouting expert en analyse football, specialise UNIQUEMENT dans la Botola Pro marocaine et le Wydad AC (WAC).

Tu recois des extraits bruts de la base de donnees FootyStats Botola Pro 2025/2026
ainsi que les effectifs des equipes depuis Wikipedia.
Ta mission est de les structurer et de produire un rapport de scouting factuel.

REGLES STRICTES :
1. Tu ne dois utiliser que les donnees fournies dans le prompt.
2. Si une information manque, dis "Non disponible" au lieu d'inventer.
3. Tu ne dois JAMAIS mentionner d'autres championnats ou equipes etrangeres.
4. Chaque fait doit etre directement issu des extraits fournis.
5. Pour les joueurs cles, utilise UNIQUEMENT les effectifs fournis.

Format de sortie :
- Forme recente du WAC (stats + resultats)
- Forme recente de l'adversaire (stats + resultats)
- Joueurs cles (avec position et nationalite, issus des effectifs fournis)
- Confrontations directes (si trouvees)
- Faits saillants
- Informations manquantes"""

    def __init__(self):
        super().__init__(
            name="Scout",
            role=self.SYSTEM_PROMPT,
            temperature=0.2,
        )
        self._index = None

    def _get_index(self):
        if self._index is None:
            self._index = build_or_load_index()
        return self._index

    def _search(self, query: str, top_k: int = 2) -> str:
        """Effectue une recherche RAG et retourne le texte formate (tronque pour eviter les limites TPM)."""
        results = retrieve_context(self._get_index(), query, top_k=top_k, min_score=0.15)
        if not results:
            return f"[Aucun resultat pertinent pour : {query}]"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"--- Source {i} (score: {r['score']:.3f}) ---")
            # Limite a 600 caracteres pour ne pas depasser les limites TPM de Groq
            text = r["text"][:600]
            if len(r["text"]) > 600:
                text += "... [tronque]"
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

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
        """Extrait le club ACTUEL depuis le champ 'club' brute de FootyStats.

        Le scraper FootyStats concatene le texte de la page. La structure est:
        'Wydad CasablancaNational Team : MoroccoPosition : ...'
        ou pour un joueur transfere:
        'CR Khemis ZemamraNational Team : Wydad CasablancaPosition : ...'
        On prend le texte avant 'National Team :' comme club actuel.
        """
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

            # Parse le club ACTUEL pour chaque joueur
            df["parsed_club"] = df["club"].apply(self._parse_footystats_club)

            # Filtre par club actuel (pas par .str.contains qui match aussi National Team)
            mask = df["parsed_club"].apply(lambda x: self._club_matches(x, club_name))
            df = df[mask].copy()

            # Filtre : garder uniquement les joueurs avec au moins 150 minutes ET 2 matchs
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
            aliases = club_aliases.get(club_name, [club_name])
            mask = df["club"].str.contains(aliases[0], case=False, na=False)
            for alias in aliases[1:]:
                mask = mask | df["club"].str.contains(alias, case=False, na=False)
            df = df[mask].copy()

            # Exclusions manuelles (anciens joueurs)
            exclusions = self._load_exclusions()
            if exclusions:
                df = df[~df["player_name"].isin(exclusions)].copy()

            # Filtre : garder uniquement les joueurs avec au moins 300 minutes ET 3 matchs
            df["mins"] = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0)
            df["mats"] = pd.to_numeric(df.get("matches_played"), errors="coerce").fillna(0)
            df = df[(df["mins"] >= 300) & (df["mats"] >= 3)].copy()
            if df.empty:
                return None

            # Parse position
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
            df["Nationality"] = ""
            # Extrait nationalite depuis la colonne position si possible
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
        """Formate un DataFrame d'effectif en texte condense pour le prompt (limite taille)."""
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
                # Garde toujours le capitaine, limite a 3 par poste
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
        # Top buteurs (max 3)
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
        # Joueurs clés (plus de minutes, max 3)
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

    def _load_team_text(self, slug: str, is_wac: bool = False) -> str:
        """Charge le fichier texte d'une equipe depuis data/raw en extrayant les sections cles."""
        if is_wac:
            path = Path("data/raw/wydad-athletic-club-2530_COMPLETE.txt")
        else:
            path = Path(f"data/raw/{slug}_stats.txt")
        if not path.exists():
            return "[Fiche non disponible]"
        text = path.read_text(encoding="utf-8")
        max_len = 3500
        if len(text) <= max_len:
            return text
        # Extrait les sections importantes par mot-cle
        sections = []
        keywords = [
            ("Stats principales", 0, 1200),
            ("BTTS", "Over / BTTS", "BTTS Stats"),
            ("Timing buts", "Scored 1st/2nd Half", "1st Half", "2nd Half"),
            ("Corners", "Match Corners", "Team Corners"),
            ("Cartons", "Match Cards", "Cards For", "Cards Against"),
            ("Fin", -800, None),
        ]
        for label, *args in keywords:
            if label == "Stats principales":
                sections.append(text[args[0]:args[1]])
            elif label == "Fin":
                sections.append(text[args[0]:])
            else:
                for kw in args:
                    idx = text.find(kw)
                    if idx != -1:
                        start = max(0, idx - 50)
                        end = min(len(text), idx + 400)
                        sections.append(text[start:end])
                        break
        return "\n\n---\n\n".join(sections)

    def scout_match(self, adversaire: str, contexte: str = "") -> Dict[str, Any]:
        """
        Effectue le scouting pour un match WAC vs adversaire.

        Args:
            adversaire: Nom de l'equipe adverse
            contexte: Informations supplementaires

        Returns:
            Resultat du scouting avec toutes les donnees brutes collectees
        """
        print(f"[Scout] Recherche des donnees pour WAC vs {adversaire}...")

        # FICHES COMPLETES chargees DIRECTEMENT (pas via RAG) pour avoir toutes les stats
        wac_text = self._load_team_text("", is_wac=True)
        adv_slug = ADVERSAIRE_SLUG_MAP.get(adversaire)
        adv_text = self._load_team_text(adv_slug) if adv_slug else "[Fiche adverse non disponible]"

        # Recherches RAG pour les donnees contextuelles
        confrontations = self._search(f"confrontations Wydad AC {adversaire}", top_k=2)
        fixtures = self._search("fixtures complete calendrier botola", top_k=1)
        upcoming = self._search("prochain match Wydad AC upcoming", top_k=1)
        infos_comp = self._search("infos competition botola stades format", top_k=1)

        # Recherche RAG unique pour les stats complementaires
        stats_extra = self._search("BTTS corners cards timing buts Wydad AC", top_k=1)

        # Extraction programmatique de la date du prochain match contre l'adversaire
        upcoming_date = "Non disponible"
        upcoming_adversaire_lines = []
        for line in upcoming.splitlines():
            if "vs" in line and any(x in line for x in [adversaire, adversaire.replace("CA", "Casablanca"), adversaire.replace("Casablanca", "CA")]):
                date_match = re.search(r"\[([^\]]+)\]", line)
                if date_match:
                    upcoming_date = date_match.group(1)
                upcoming_adversaire_lines.append(line.strip())

        # Stats individuelles FootyStats
        wac_stats = self._load_player_stats("Wydad AC")
        adv_stats = self._load_player_stats(adversaire)
        wac_stats_fmt = self._format_player_stats(wac_stats)
        adv_stats_fmt = self._format_player_stats(adv_stats)

        # Effectifs (FootyStats filtre, plus fiable que Wikipedia seul)
        wac_squad = self._load_squad_footystats("Wydad AC")
        adv_squad = self._load_squad_footystats(adversaire)
        wac_squad_fmt = self._format_squad(wac_squad)
        adv_squad_fmt = self._format_squad(adv_squad)

        # Compilation du prompt pour le LLM
        data_block = f"""
=== FICHE COMPLETE WYDAD AC ===
{wac_text}

=== STATS COMPLEMENTAIRES WAC (RAG) ===
{stats_extra}

=== FICHE COMPLETE {adversaire.upper()} ===
{adv_text}

=== EFFECTIF WYDAD AC (joueurs avec >=300min en 2025/2026) ===
{wac_squad_fmt}

=== STATS INDIVIDUELLES JOUEURS WYDAD AC ===
{wac_stats_fmt}

=== EFFECTIF {adversaire.upper()} (joueurs avec >=300min en 2025/2026) ===
{adv_squad_fmt}

=== STATS INDIVIDUELLES JOUEURS {adversaire.upper()} ===
{adv_stats_fmt}

=== CONFRONTATIONS DIRECTES ===
{confrontations}

=== PROCHAIN(S) MATCH(S) WYDAD AC ===
{upcoming}

=== CALENDRIER COMPLET ===
{fixtures}

=== INFOS COMPETITION ===
{infos_comp}
"""

        prompt = f"""Tu es l'Agent de Scouting. Voici les donnees brutes collectees depuis la base FootyStats Botola Pro 2025/2026 pour le match WAC vs {adversaire}.

CONTEXTE : {contexte}

DATE EXTRAITE DU CALENDRIER POUR CE MATCH : {upcoming_date}
(LIGNES DU CALENDRIER CONCERNANT CE MATCH : {upcoming_adversaire_lines})

DONNEES BRUTES :
{data_block}

Produis un rapport de scouting structure contenant UNIQUEMENT les faits issus des donnees ci-dessus :

# RAPPORT DE SCOUTING - WAC vs {adversaire}

## 1. Contexte du Match
- Competition: Botola Pro 2025/2026
- Journee: (deduis du calendrier si possible, sinon "Non specifiee")
- Date et heure: {upcoming_date} (utilise cette date extraite du calendrier, ou "Non disponible" si vide)
- Stade: (utilise les infos de competition si disponible, sinon "Non disponible")
- Format: 16 equipes, 30 journees
- Classement WAC: (position approximative si deduisible)
- Classement adversaire: (position approximative si deduisible)

## 2. Forme du WAC (Stats detaillees)
(Resumer les stats collectives du WAC: Wins%, xG, buts marques/encaisses, possession, clean sheets, etc.)
(Utilise les stats des tableaux completes: BTTS%, corners, cartons, timing des buts, etc.)

## 3. Forme de {adversaire} (Stats detaillees)
(Resumer les stats collectives de l'adversaire si disponibles)

## 4. Confrontations Directes WAC vs {adversaire}
(Historique des matchs: scores, dates, resultats - uniquement si trouves dans les donnees)
(Si des matchs sont a venir, indique la date prevue)

## 5. Analyse des Tendances
- BTTS (Both Teams To Score) pour chaque equipe
- Over/Under 2.5 buts
- Corners moyens
- Cartons moyens
- Timing des buts (1ere/2eme mi-temps)
- Performance domicile vs exterieur
- Goal kicks, throw-ins, free kicks

## 6. Joueurs Cles (OBLIGATOIRE - utilise les effectifs ET les stats individuelles)
UTILISE IMPERATIVEMENT les effectifs fournis dans les sections "EFFECTIF WYDAD AC" et "EFFECTIF ADVERSAIRE".
Tu DOIS citer les noms reels des joueurs avec leur position.
- WAC: liste le capitaine (Nordin Amrabat si present) et 3-4 joueurs cles
- Adversaire: liste le(s) capitaine(s) et 3-4 joueurs cles
- N'invente PAS de joueurs qui ne sont pas dans les effectifs fournis
- Si les effectifs sont vides ou absents, dis "Effectif non disponible dans la base"
- CROISE avec les "STATS INDIVIDUELLES JOUEURS" : privilegie les joueurs avec le plus de buts, d'assists, de minutes ou la meilleure note. Mentionne leurs stats chiffrees (buts, xG, passes D, xA, minutes, rating) quand elles sont disponibles.

## 7. Faits Saillants
(Blessures, suspensions, tendances - uniquement si presents)

## 8. Informations Manquantes
(Liste ce qui n'a pas ete trouve dans les donnees)

        RAPPEL CRITIQUE :
        - Base-toi UNIQUEMENT sur les donnees fournies.
        - Ne rajoute RIEN d'exterieur.
        - Si une info n'est pas dans les donnees, dis "Non disponible"."""

        result = self.run(prompt)

        # Post-processing : injection programmatique de la date si l'LLM ne l'a pas utilisee
        if upcoming_date != "Non disponible" and "output" in result:
            output = result["output"]
            # Remplace les patterns "Non disponible" apres "Date et heure:" par la vraie date
            output = re.sub(
                r"(Date et heure:\s*)Non disponible",
                rf"\g<1>{upcoming_date}",
                output,
                flags=re.IGNORECASE,
            )
            result["output"] = output

        return result
