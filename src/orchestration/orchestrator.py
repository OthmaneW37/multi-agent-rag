"""Orchestrateur principal du systeme multi-agents WAC Sport Analytics."""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.agents.scout import ScoutAgent
from src.agents.modelisateur import ModelisateurAgent
from src.agents.tacticien import TacticienAgent
from src.agents.validateur import ValidateurAgent
from src.orchestration.match_context import get_match_context


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


def _inject_context_in_report(report: str, ctx: Dict[str, str]) -> str:
    """Injecte le contexte de match enrichi (date, journee, lieu, enjeux) dans le rapport."""
    # Competition / Journee
    if ctx.get("journee") and ctx["journee"] != "Non disponible":
        report = re.sub(
            r"(Journee\s*[:\-]?\s*)(Non disponible|Inconnue|TBD)",
            rf"\g<1>{ctx['journee']}",
            report,
            flags=re.IGNORECASE,
        )
    # Lieu
    if ctx.get("lieu") and ctx["lieu"] != "Non disponible":
        report = re.sub(
            r"(Lieu\s*[:\-]?\s*)(Non disponible|Inconnue|TBD)",
            rf"\g<1>{ctx['lieu']}",
            report,
            flags=re.IGNORECASE,
        )
    # Date
    if ctx.get("date") and ctx["date"] != "Non disponible":
        report = re.sub(
            r"(Date\s*[:\-]?\s*)(Non disponible|Inconnue|TBD)",
            rf"\g<1>{ctx['date']}",
            report,
            flags=re.IGNORECASE,
        )
        # Aussi remplacer dans le resume executif si present
        report = re.sub(
            r"(Date et heure\s*[:\-]?\s*)Non disponible",
            rf"\g<1>{ctx['date']}",
            report,
            flags=re.IGNORECASE,
        )
    # Enjeux
    if ctx.get("enjeux") and ctx["enjeux"] != "Non disponible":
        # Cherche la ligne "Enjeux : Non disponible" ou similaire
        report = re.sub(
            r"(Enjeux\s*[:\-]?\s*)(Non disponible|Inconnue|TBD).*?(\n|$)",
            rf"\g<1>{ctx['enjeux']}\n",
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


def _verifier_qualite_scout(donnees_scout: str) -> tuple[bool, str]:
    """
    Verifie que le scouting a produit des donnees exploitables.

    Retourne (ok, message) où ok est True si la qualite est suffisante.
    """
    if len(donnees_scout) < 200:
        return False, "Le rapport de scouting est trop court (< 200 caracteres)."

    missing_count = donnees_scout.lower().count("aucun resultat")
    if missing_count >= 5:
        return False, "Le Scout n'a trouve aucune donnee pertinente dans la base RAG."

    # Si le rapport est substantiel (>400 chars) et contient peu d'echecs RAG,
    # on accepte meme sans ## (le Scout tente un reformatage en amont)
    if len(donnees_scout) >= 400 and missing_count < 5:
        return True, ""

    # Se bloquer sur le format ## uniquement si le texte est vraiment court/maigre
    if len(donnees_scout) < 400 and "##" not in donnees_scout:
        return False, "Le rapport de scouting ne contient pas de sections structurees et est trop court."

    return True, ""


class WACOrchestrator:
    """
    Orchestrateur qui coordonne les 4 agents du pipeline d'analyse.

    Flux de travail :
    1. ScoutAgent -> Collecte les donnees brutes via RAG (ReAct + outils)
    2. ModelisateurAgent -> Analyse les donnees et genere des predictions
    3. TacticienAgent -> Formule des recommandations strategiques
    4. ValidateurAgent -> Valide et compile le rapport final

    Routage conditionnel :
    - Si le Scout ne trouve pas assez de donnees, le pipeline s'arrete
      et retourne un message d'erreur explicite.
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
        on_event=None,
    ) -> Dict[str, Any]:
        """
        Execute le pipeline complet d'analyse pour un match WAC vs adversaire.

        Args:
            adversaire: Nom de l'equipe adverse
            contexte: Informations supplementaires sur le match
            verbose: Affiche les etapes intermediaires
            on_event: Callback optionnel(event_type, agent_id, data) pour streaming

        Returns:
            Dictionnaire contenant tous les resultats du pipeline
        """
        def _emit(event_type, agent_id, data=None):
            if on_event:
                on_event(event_type, agent_id, data)

        results = {
            "adversaire": adversaire,
            "contexte": contexte,
            "etapes": {},
        }

        # Etape 1 : Scouting
        _emit("agent_start", "scout")
        if verbose:
            print(f"\n{'='*60}")
            print(f"ETAPE 1/4 : SCOUTING - Collecte de donnees sur {adversaire}")
            print(f"{'='*60}\n")

        scout_result = self.scout.scout_match(adversaire, contexte)
        results["etapes"]["scout"] = scout_result
        donnees_scout = scout_result.get("output", str(scout_result))

        _emit("agent_message", "scout", {"output": donnees_scout})

        if verbose:
            print(f"\n[SCOUT OUTPUT]\n{donnees_scout[:2000]}\n... [truncated]\n")

        # Routage conditionnel : verifier la qualite du scouting
        scout_ok, scout_msg = _verifier_qualite_scout(donnees_scout)
        if not scout_ok:
            print(f"\n[ORCHESTRATEUR] ARRET DU PIPELINE : {scout_msg}")
            _emit("agent_error", "scout", {"message": scout_msg})
            results["erreur"] = scout_msg
            results["rapport_final"] = (
                f"# ERREUR - Analyse impossible\n\n"
                f"Le pipeline s'est arrete a l'etape de scouting.\n\n"
                f"**Raison** : {scout_msg}\n\n"
                f"**Conseil** : Verifiez que les donnees FootyStats sont presentes "
                f"dans `data/raw/` et que l'index RAG est construit."
            )
            _emit("pipeline_end", "orchestrateur", results)
            return results

        _emit("agent_end", "scout")

        # Etape 2 : Modelisation
        _emit("agent_start", "modelisateur")
        if verbose:
            print(f"\n{'='*60}")
            print(f"ETAPE 2/4 : MODELISATION - Analyse des donnees")
            print(f"{'='*60}\n")

        modelisateur_result = self.modelisateur.analyser(
            donnees_scout, adversaire
        )
        results["etapes"]["modelisateur"] = modelisateur_result
        analyse_modelisateur = modelisateur_result.get("output", str(modelisateur_result))

        _emit("agent_message", "modelisateur", {"output": analyse_modelisateur})

        if verbose:
            print(f"\n[MODELISATEUR OUTPUT]\n{analyse_modelisateur[:2000]}\n... [truncated]\n")
            print(f"\n{'='*60}")
            print(f"ETAPE 3/4 : TACTIQUE - Formulation de la strategie")
            print(f"{'='*60}\n")

        _emit("agent_end", "modelisateur")

        # Etape 3 : Tactique
        _emit("agent_start", "tacticien")
        analyse_truncated = analyse_modelisateur[:1500] + "\n... [analyse tronquee pour limites de taille]\n" if len(analyse_modelisateur) > 1500 else analyse_modelisateur
        tacticien_result = self.tacticien.formuler_strategie(
            analyse_truncated, adversaire
        )
        results["etapes"]["tacticien"] = tacticien_result
        strategie_tacticien = tacticien_result.get("output", str(tacticien_result))

        _emit("agent_message", "tacticien", {"output": strategie_tacticien})

        if verbose:
            print(f"\n[TACTICIEN OUTPUT]\n{strategie_tacticien[:2000]}\n... [truncated]\n")
            print(f"\n{'='*60}")
            print(f"ETAPE 4/4 : VALIDATION - Compilation du rapport final")
            print(f"{'='*60}\n")

        _emit("agent_end", "tacticien")

        # Etape 4 : Validation et rapport final
        _emit("agent_start", "validateur")
        validateur_result = self.validateur.valider_et_compiler(
            donnees_scout,
            analyse_modelisateur,
            strategie_tacticien,
            adversaire,
        )
        results["etapes"]["validateur"] = validateur_result
        rapport_final = validateur_result.get("output", str(validateur_result))

        _emit("agent_message", "validateur", {"output": rapport_final})

        # Post-processing : injection du contexte de match enrichi
        ctx = get_match_context(adversaire)
        rapport_final = _inject_context_in_report(rapport_final, ctx)

        # Fallback anciens extracteurs si les nouveaux n'ont pas tout trouve
        upcoming_date = _extract_upcoming_match_date(adversaire)
        stadium = _extract_stadium(adversaire)
        confrontations = _extract_confrontations(adversaire)
        rapport_final = _inject_date_in_report(rapport_final, upcoming_date)
        rapport_final = _inject_stadium_in_report(rapport_final, stadium)
        rapport_final = _inject_confrontations_in_report(rapport_final, confrontations)

        xi_text = _generate_xi_programmatic("Wydad AC")
        rapport_final = _inject_xi_in_report(rapport_final, xi_text)

        results["rapport_final"] = rapport_final

        _emit("agent_end", "validateur")
        _emit("pipeline_end", "orchestrateur", results)

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
