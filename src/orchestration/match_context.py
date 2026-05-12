"""Module d'enrichissement du contexte de match.

Lit les fixtures FootyStats et les stats collectives pour generer
un contexte de match riche : date, journee, lieu (domicile/exterieur), enjeux.
"""

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRAP_PATH = Path("ScrappingDataBotola/data")
TEAMS_PATH = SCRAP_PATH / "teams"

CLUB_ALIASES = {
    "Wydad AC": ["wydad casablanca", "wydad athletic club"],
    "Raja CA": ["raja casablanca", "raja club athletic"],
    "FAR Rabat": ["far rabat", "forces armees royales"],
    "FUS Rabat": ["fus rabat", "fath union sport"],
    "Ittihad Tanger": ["ittihad tanger", "al-ittihad"],
    "Olympic Safi": ["olympic safi", "olympique safi"],
    "Hassania Agadir": ["hassania agadir", "hus agadir"],
    "Maghreb Fes": ["maghreb fes", "mas fes", "maghreb fez"],
    "Kawkab Marrakech": ["kawkab marrakech", "kac marrakech"],
    "RSB Berkane": ["rsb berkane", "rs berkane", "renaissance sportive"],
    "DH El Jadida": ["difaa hassani", "dh el jadida", "difaâ el jadida"],
    "CODM Meknes": ["codm meknes", "club omnisports de meknes"],
    "Yacoub El Mansour": ["yacoub el mansour", "us yacoub"],
    "UTS Rabat": ["uts rabat"],
    "Olympique Dcheira": ["olympique dcheira", "olympique dcheïra"],
    "CR Khemis Zemamra": ["cr khemis zemamra", "club renaissance khemis"],
}


def _matches_team(team_name: str, raw_name: str) -> bool:
    raw_lower = raw_name.lower()
    aliases = CLUB_ALIASES.get(team_name, [team_name.lower()])
    return any(alias in raw_lower for alias in aliases)


def _read_fixtures() -> List[Dict[str, str]]:
    filepath = SCRAP_PATH / "fixtures.csv"
    if not filepath.exists():
        return []
    rows = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_team_stats(team_name: str) -> Dict[str, str]:
    """Lit les stats collectives d'un club depuis le CSV team."""
    # Mapping nom -> slug
    slug_map = {
        "Wydad AC": "wydad-athletic-club-2530",
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
    slug = slug_map.get(team_name)
    if not slug:
        return {}
    filepath = TEAMS_PATH / f"{slug}.csv"
    if not filepath.exists():
        return {}
    stats = {}
    try:
        with open(filepath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stat = row.get("Stats", "").strip()
                val = row.get("Overall", "").strip()
                if stat:
                    stats[stat] = val
    except Exception:
        pass
    return stats


def get_match_context(adversaire: str) -> Dict[str, str]:
    """
    Genere le contexte enrichi d'un match WAC vs adversaire.

    Retourne un dict avec :
        - competition : toujours "Botola Pro 2025/2026"
        - journee : numero estime
        - lieu : "Stade Mohamed V (Casablanca)" si domicile, sinon "Deplacement"
        - date : date du match depuis fixtures.csv
        - enjeux : texte genere depuis les stats
        - is_home : "oui" ou "non"
        - wac_forme : bilan WAC
        - adv_forme : bilan adversaire
    """
    fixtures = _read_fixtures()
    wac_aliases = ["wydad casablanca", "wydad athletic club", "wydad ac"]
    adv_aliases = CLUB_ALIASES.get(adversaire, [adversaire.lower()])

    # Chercher le prochain match WAC vs adversaire
    upcoming_match = None
    played_matches_wac = []
    played_matches_adv = []

    for fx in fixtures:
        home = fx.get("Home", "").lower()
        away = fx.get("Away", "").lower()
        score = fx.get("Score", "").strip()
        date_str = fx.get("Date", "").strip()
        status = fx.get("Status", "").strip()

        is_wac = any(a in home or a in away for a in wac_aliases)
        is_adv = any(a in home or a in away for a in adv_aliases)

        if is_wac:
            if score and " - " in score:
                played_matches_wac.append(fx)
            elif not score and not upcoming_match:
                # Premier match sans score = prochain match
                upcoming_match = fx

        if is_adv and score and " - " in score:
            played_matches_adv.append(fx)

    # Journee = nombre de matchs joues + 1
    journee = len(played_matches_wac) + 1

    # Date
    date_raw = upcoming_match.get("Date", "") if upcoming_match else ""
    date_clean = "Non disponible"
    if date_raw:
        # Essayer de parser "May 9, 8:00pm"
        m = re.match(r"([A-Za-z]+)\s+(\d+),\s*(\d+:\d+[ap]m)", date_raw)
        if m:
            date_clean = f"{m.group(1)} {m.group(2)} — {m.group(3)}"
        else:
            date_clean = date_raw

    # Domicile / Exterieur
    is_home = False
    if upcoming_match:
        home_raw = upcoming_match.get("Home", "").lower()
        is_home = any(a in home_raw for a in wac_aliases)

    lieu = "Stade Mohamed V (Casablanca)" if is_home else f"Deplacement — Stade de {adversaire}"

    # Stats pour enjeux
    wac_stats = _read_team_stats("Wydad AC")
    adv_stats = _read_team_stats(adversaire)

    wac_wins = wac_stats.get("Wins", "N/A")
    adv_wins = adv_stats.get("Wins", "N/A")
    wac_poss = wac_stats.get("Possession AVG", "N/A")
    adv_poss = adv_stats.get("Possession AVG", "N/A")
    wac_xg = wac_stats.get("xG For / Match", "N/A")
    adv_xg = adv_stats.get("xG For / Match", "N/A")

    # Enjeux dynamique
    enjeux_parts = []
    if is_home:
        enjeux_parts.append("Match a domicile au Stade Mohamed V. Le WAC doit capitaliser sur l'effet de ses supporters.")
    else:
        enjeux_parts.append(f"Match a l'exterieur contre {adversaire}. Le WAC devra compter sur sa solidite defensive.")

    if wac_wins != "N/A" and adv_wins != "N/A":
        try:
            w_w = float(wac_wins.replace("%", ""))
            a_w = float(adv_wins.replace("%", ""))
            if w_w > a_w:
                enjeux_parts.append(f"Le WAC affiche un meilleur ratio de victoires ({wac_wins}) que {adversaire} ({adv_wins}).")
            elif a_w > w_w:
                enjeux_parts.append(f"{adversaire} presente un ratio de victoires superieur ({adv_wins}) a celui du WAC ({wac_wins}).")
            else:
                enjeux_parts.append(f"Les deux equipes sont a egalite en termes de victoires ({wac_wins}).")
        except ValueError:
            pass

    if wac_poss != "N/A" and adv_poss != "N/A":
        enjeux_parts.append(f"Possession moyenne : WAC {wac_poss} vs {adversaire} {adv_poss}.")

    if wac_xg != "N/A" and adv_xg != "N/A":
        try:
            w_xg = float(wac_xg)
            a_xg = float(adv_xg)
            if w_xg > a_xg:
                enjeux_parts.append(f"Le WAC genere plus de xG par match ({wac_xg}) que {adversaire} ({adv_xg}).")
            else:
                enjeux_parts.append(f"{adversaire} genere plus de xG par match ({adv_xg}) que le WAC ({wac_xg}).")
        except ValueError:
            pass

    enjeux = " ".join(enjeux_parts) if enjeux_parts else "Match crucial pour le classement de Botola Pro."

    return {
        "competition": "Botola Pro 2025/2026",
        "journee": f"J{journee}" if journee > 0 else "Non disponible",
        "lieu": lieu,
        "date": date_clean,
        "enjeux": enjeux,
        "is_home": "oui" if is_home else "non",
        "wac_forme": f"Victoires: {wac_wins}" if wac_wins != "N/A" else "Stats non disponibles",
        "adv_forme": f"Victoires: {adv_wins}" if adv_wins != "N/A" else "Stats non disponibles",
    }
