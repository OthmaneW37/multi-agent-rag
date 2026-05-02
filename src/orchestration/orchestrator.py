"""Orchestrateur principal du systeme multi-agents WAC Sport Analytics."""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.agents.scout import ScoutAgent
from src.agents.modelisateur import ModelisateurAgent
from src.agents.tacticien import TacticienAgent
from src.agents.validateur import ValidateurAgent


def _generate_xi_programmatic(club_name: str = "Wydad AC") -> str:
    """Genere un XI type programmatiquement depuis les stats FootyStats."""
    path = Path("ScrappingDataBotola/data/player_stats.csv")
    if not path.exists():
        return "[Fichier player_stats.csv non trouve]"
    try:
        df = pd.read_csv(path)
        if df.empty or "club" not in df.columns:
            return "[Donnees vides]"

        def _parse_club(club_raw):
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

        def _club_matches(parsed, target):
            parsed_lower = parsed.lower()
            aliases = {
                "wydad ac": ["wydad", "wydad casablanca"],
            }
            return any(a in parsed_lower for a in aliases.get(target.lower(), [target.lower()]))

        df["parsed_club"] = df["club"].apply(_parse_club)
        df = df[df["parsed_club"].apply(lambda x: _club_matches(x, club_name))].copy()

        if df.empty:
            return "[Aucun joueur trouve pour ce club]"

        df["mins_num"] = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0)
        df["rating_num"] = pd.to_numeric(df.get("rating"), errors="coerce").fillna(0)
        df["goals_num"] = pd.to_numeric(df.get("goals_scored"), errors="coerce").fillna(0)

        def _parse_pos(pos_str):
            if pd.isna(pos_str):
                return ""
            m = re.match(r"^(Goalkeeper|Defender|Midfielder|Forward)(?:\s*-.*)?", str(pos_str), re.IGNORECASE)
            if m:
                mapping = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Forward": "FW"}
                return mapping.get(m.group(1).capitalize(), m.group(1).capitalize())
            return ""

        df["pos"] = df["position"].apply(_parse_pos)
        lines = []

        gk = df[df["pos"] == "GK"].nlargest(1, "mins_num")
        if not gk.empty:
            r = gk.iloc[0]
            lines.append(f"- Gardien: {r['player_name']} ({r['minutes']}min, note {r['rating']})")
        else:
            lines.append("- Gardien: [Non disponible]")

        dfs = df[df["pos"] == "DF"].copy()
        if not dfs.empty:
            dfs["score"] = dfs["mins_num"] * 0.7 + dfs["rating_num"] * 10
            dfs = dfs.nlargest(4, "score")
            names = [f"{r['player_name']} ({r['minutes']}min, n:{r['rating']})" for _, r in dfs.iterrows()]
            lines.append("- Defenseurs: " + ", ".join(names))
        else:
            lines.append("- Defenseurs: [Non disponibles]")

        mfs = df[df["pos"] == "MF"].copy()
        if not mfs.empty:
            mfs["score"] = mfs["mins_num"] * 0.6 + mfs["goals_num"] * 50 + mfs["rating_num"] * 10
            mfs = mfs.nlargest(3, "score")
            names = [f"{r['player_name']} ({r['minutes']}min, n:{r['rating']}, {r['goals_scored']}B)" for _, r in mfs.iterrows()]
            lines.append("- Milieux: " + ", ".join(names))
        else:
            lines.append("- Milieux: [Non disponibles]")

        fws = df[df["pos"] == "FW"].copy()
        if not fws.empty:
            fws["score"] = fws["mins_num"] * 0.5 + fws["goals_num"] * 100 + fws["rating_num"] * 10
            fws = fws.nlargest(3, "score")
            names = [f"{r['player_name']} ({r['minutes']}min, n:{r['rating']}, {r['goals_scored']}B)" for _, r in fws.iterrows()]
            lines.append("- Attaquants: " + ", ".join(names))
        else:
            lines.append("- Attaquants: [Non disponibles]")

        return "\n".join(lines)
    except Exception as exc:
        return f"[Erreur generation XI: {exc}]"


def _extract_upcoming_match_date(adversaire: str) -> str:
    """Extrait la date du prochain match WAC vs adversaire depuis le fichier dedie."""
    filepath = Path("data/raw/prochain_match_wac.txt")
    if not filepath.exists():
        return "Non disponible"

    text = filepath.read_text(encoding="utf-8")
    # Cherche une ligne contenant l'adversaire (accepte Raja CA / Raja Casablanca)
    aliases = [adversaire, adversaire.replace("CA", "Casablanca"), adversaire.replace("Casablanca", "CA")]
    for line in text.splitlines():
        if "vs" in line and any(alias in line for alias in aliases):
            match = re.search(r"\[([^\]]+)\]", line)
            if match:
                return match.group(1)
    return "Non disponible"


def _extract_stadium(adversaire: str) -> str:
    """Extrait le stade depuis infos_competition.txt si disponible."""
    filepath = Path("data/raw/infos_competition.txt")
    if not filepath.exists():
        return "Non disponible"

    text = filepath.read_text(encoding="utf-8")
    # Cherche le stade de l'adversaire en priorite, puis WAC en fallback
    aliases_priority = [adversaire, adversaire.replace("CA", "Casablanca"), adversaire.replace("Casablanca", "CA")]
    aliases_fallback = ["Wydad AC"]

    for aliases in [aliases_priority, aliases_fallback]:
        for line in text.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("- ") and ":" in line_stripped:
                for alias in aliases:
                    if alias in line_stripped:
                        parts = line_stripped.split(":", 1)
                        if len(parts) == 2:
                            return parts[1].strip()
    return "Non disponible"


def _inject_date_in_report(report: str, date: str) -> str:
    """Remplace 'Non disponible' apres 'Date et heure:' par la vraie date."""
    if date == "Non disponible":
        return report
    report = re.sub(
        r"(Date et heure\s*:\s*)Non disponible",
        rf"\g<1>{date}",
        report,
        flags=re.IGNORECASE,
    )
    return report


def _inject_stadium_in_report(report: str, stadium: str) -> str:
    """Remplace 'Non disponible' apres 'Stade:' par le vrai stade."""
    if stadium == "Non disponible":
        return report
    report = re.sub(
        r"(Stade\s*:\s*)Non disponible",
        rf"\g<1>{stadium}",
        report,
        flags=re.IGNORECASE,
    )
    return report


def _inject_xi_in_report(report: str, xi_text: str) -> str:
    """Injecte le XI programmatique dans la section Composition du rapport final."""
    if not xi_text or xi_text.startswith("["):
        return report
    # Remplace la section Composition existante par le XI programmatique
    report = re.sub(
        r"(## 5\. Recommandations Strategiques.*?- Composition probable:).*?(\n- Consignes cles|\n## 6\.)",
        rf"\g<1>\n{xi_text}\g<2>",
        report,
        flags=re.DOTALL,
    )
    # Si pas de remplacement, ajoute apres Schema tactique
    if "Gardien:" not in report:
        report = re.sub(
            r"(Schema tactique conseille:.*?\n)",
            rf"\g<1>- Composition probable (generee depuis FootyStats):\n{xi_text}\n",
            report,
            flags=re.DOTALL,
        )
    return report


def _extract_confrontations(adversaire: str) -> str:
    """Extrait les confrontations WAC vs adversaire depuis le fichier dedie."""
    filepath = Path("data/raw/confrontations_directes_wac.txt")
    if not filepath.exists():
        return "Non disponible"

    text = filepath.read_text(encoding="utf-8")
    aliases = [adversaire, adversaire.replace("CA", "Casablanca"), adversaire.replace("Casablanca", "CA")]

    in_section = False
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--- Wydad AC vs"):
            in_section = any(alias in stripped for alias in aliases)
            continue
        if in_section:
            if stripped.startswith("--- Wydad AC vs") or stripped.startswith("### "):
                break
            if stripped and not stripped.startswith("(Aucun match"):
                lines.append(stripped)

    if lines:
        return "; ".join(lines)
    return "Non disponible"


def _inject_confrontations_in_report(report: str, confrontations: str) -> str:
    """Remplace le contenu apres 'Confrontations directes:' par les vraies confrontations."""
    if confrontations == "Non disponible":
        return report
    # Remplace tout ce qui suit "Confrontations directes:" sur la meme ligne
    report = re.sub(
        r"(Confrontations directes\s*:\s*)(?!\s*$).*",
        rf"\g<1>{confrontations}",
        report,
        flags=re.IGNORECASE,
    )
    return report


class WACOrchestrator:
    """
    Orchestrateur qui coordonne les 4 agents du pipeline d'analyse :

    1. ScoutAgent -> Collecte les donnees brutes via RAG
    2. ModelisateurAgent -> Analyse les donnees et genere des predictions
    3. TacticienAgent -> Formule des recommandations strategiques
    4. ValidateurAgent -> Valide et compile le rapport final
    """

    def __init__(self):
        self.scout = ScoutAgent()
        self.modelisateur = ModelisateurAgent()
        self.tacticien = TacticienAgent()
        self.validateur = ValidateurAgent()

    def analyser_match(
        self,
        adversaire: str,
        contexte: str = "",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute le pipeline complet d'analyse pour un match WAC vs adversaire.

        Args:
            adversaire: Nom de l'equipe adverse
            contexte: Informations supplementaires sur le match
            verbose: Affiche les etapes intermediaires

        Returns:
            Dictionnaire contenant tous les resultats du pipeline
        """
        results = {
            "adversaire": adversaire,
            "contexte": contexte,
            "etapes": {},
        }

        # Etape 1 : Scouting
        if verbose:
            print(f"\n{'='*60}")
            print(f"ETAPE 1/4 : SCOUTING - Collecte de donnees sur {adversaire}")
            print(f"{'='*60}\n")

        scout_result = self.scout.scout_match(adversaire, contexte)
        results["etapes"]["scout"] = scout_result
        donnees_scout = scout_result.get("output", str(scout_result))

        if verbose:
            print(f"\n[SCOUT OUTPUT]\n{donnees_scout[:2000]}\n... [truncated]\n")
            print(f"\n{'='*60}")
            print(f"ETAPE 2/4 : MODELISATION - Analyse des donnees")
            print(f"{'='*60}\n")

        # Etape 2 : Modelisation
        modelisateur_result = self.modelisateur.analyser(
            donnees_scout, adversaire
        )
        results["etapes"]["modelisateur"] = modelisateur_result
        analyse_modelisateur = modelisateur_result.get("output", str(modelisateur_result))

        if verbose:
            print(f"\n[MODELISATEUR OUTPUT]\n{analyse_modelisateur[:2000]}\n... [truncated]\n")
            print(f"\n{'='*60}")
            print(f"ETAPE 3/4 : TACTIQUE - Formulation de la strategie")
            print(f"{'='*60}\n")

        # Etape 3 : Tactique
        # Tronque l'analyse du Modelisateur pour ne pas depasser les limites TPM
        analyse_truncated = analyse_modelisateur[:1500] + "\n... [analyse tronquee pour limites de taille]\n" if len(analyse_modelisateur) > 1500 else analyse_modelisateur
        tacticien_result = self.tacticien.formuler_strategie(
            analyse_truncated, adversaire
        )
        results["etapes"]["tacticien"] = tacticien_result
        strategie_tacticien = tacticien_result.get("output", str(tacticien_result))

        if verbose:
            print(f"\n[TACTICIEN OUTPUT]\n{strategie_tacticien[:2000]}\n... [truncated]\n")
            print(f"\n{'='*60}")
            print(f"ETAPE 4/4 : VALIDATION - Compilation du rapport final")
            print(f"{'='*60}\n")

        # Etape 4 : Validation et rapport final
        validateur_result = self.validateur.valider_et_compiler(
            donnees_scout,
            analyse_modelisateur,
            strategie_tacticien,
            adversaire,
        )
        results["etapes"]["validateur"] = validateur_result
        rapport_final = validateur_result.get("output", str(validateur_result))

        # Post-processing : injection de la date, du stade et des confrontations si trouves
        upcoming_date = _extract_upcoming_match_date(adversaire)
        stadium = _extract_stadium(adversaire)
        confrontations = _extract_confrontations(adversaire)
        rapport_final = _inject_date_in_report(rapport_final, upcoming_date)
        rapport_final = _inject_stadium_in_report(rapport_final, stadium)
        rapport_final = _inject_confrontations_in_report(rapport_final, confrontations)

        # Injection du XI genere programmatiquement
        xi_text = _generate_xi_programmatic("Wydad AC")
        rapport_final = _inject_xi_in_report(rapport_final, xi_text)

        results["rapport_final"] = rapport_final

        if verbose:
            print(f"\n{'='*60}")
            print(f"RAPPORT FINAL - WAC vs {adversaire}")
            print(f"{'='*60}\n")
            print(rapport_final)

        return results

    def analyser_match_rapide(
        self,
        adversaire: str,
        contexte: str = "",
    ) -> str:
        """
        Version simplifiee qui retourne uniquement le rapport final.

        Args:
            adversaire: Nom de l'equipe adverse
            contexte: Informations supplementaires sur le match

        Returns:
            Rapport final en Markdown
        """
        results = self.analyser_match(
            adversaire, contexte, verbose=False
        )
        return results.get("rapport_final", "Aucun rapport genere.")
