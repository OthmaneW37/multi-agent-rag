# -*- coding: utf-8 -*-
"""
wikipedia_scraper/scraper.py
----------------------------
Scraper des effectifs depuis Wikipedia EN pour la Botola Pro 2025/2026.

Récupère pour chaque club :
    - Numéro
    - Position (GK, DF, MF, FW)
    - Nationalité
    - Nom du joueur
    - Âge (si disponible)
    - Capitaine (si indiqué)

Usage :
    python -m wikipedia_scraper.scraper

Livrables :
    data/squads_wiki/{club_slug}.csv  -> 1 fichier par club
    data/squads_wiki/all_players.csv  -> Tous les joueurs consolidés
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

SEASON_URL = "https://en.wikipedia.org/wiki/2025%E2%80%9326_Botola"
REQUEST_DELAY = 1.5  # secondes entre requêtes (politesse)

# Mapping des noms de clubs vers les slugs FootyStats (pour compatibilité)
CLUB_SLUG_MAP = {
    "Wydad AC": "wydad-athletic-club-2530",
    "Raja CA": "raja-club-athletic-de-casablanca-2536",
    "AS FAR": "as-forces-armees-royales-de-rabat-2532",
    "FUS Rabat": "fath-union-sport-de-rabat-2543",
    "IR Tanger": "ittihad-riadhi-de-tanger-2529",
    "OC Safi": "olympique-club-de-safi-2534",
    "HUS Agadir": "hassania-union-sport-dagadir-2531",
    "MAS Fes": "maghreb-as-de-fes-2547",
    "KAC Marrakech": "kawkab-athletique-club-de-marrakech-2537",
    "RS Berkane": "renaissance-sportive-de-berkane-2541",
    "DH El-Jadida": "difaa-hassani-del-jadida-2538",
    "COD Meknès": "club-omnisports-de-meknes-2563",
    "US Yacoub El Mansour": "us-yacoub-el-mansour-1322394",
    "UTS Rabat": "uts-rabat-17956",
    "Olympique Dcheira": "olympique-dcheira-2555",
    "RCA Zemamra": "club-renaissance-khemis-zemamra-10556",
}


def fetch_url(url: str) -> Optional[str]:
    """Récupère le HTML d'une URL avec gestion d'erreurs."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.error(f"Échec requête {url}: {exc}")
        return None


def extract_clubs_from_season_page(html: str) -> list[dict]:
    """Extrait la liste des clubs depuis la page saison Wikipedia."""
    soup = BeautifulSoup(html, "lxml")
    clubs = []

    for table in soup.find_all("table", class_="wikitable"):
        headers_list = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Club" in headers_list and "City" in headers_list:
            for row in table.find_all("tr")[1:]:
                tds = row.find_all(["td", "th"])
                if not tds:
                    continue

                first_td = tds[0]
                club_name = first_td.get_text(strip=True)

                wiki_url = None
                for a in first_td.find_all("a", href=True):
                    href = a.get("href", "")
                    if href.startswith("/wiki/"):
                        wiki_url = "https://en.wikipedia.org" + href
                        break

                city = tds[1].get_text(strip=True) if len(tds) > 1 else ""
                stadium = tds[2].get_text(strip=True) if len(tds) > 2 else ""
                capacity = tds[3].get_text(strip=True) if len(tds) > 3 else ""

                clubs.append({
                    "name": club_name,
                    "wiki_url": wiki_url,
                    "city": city,
                    "stadium": stadium,
                    "capacity": capacity,
                })
            break

    logger.info(f"{len(clubs)} club(s) extraits de la page saison.")
    return clubs


def _find_squad_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """Trouve le tableau d'effectif dans la page d'un club."""
    # Stratégie 1: chercher un tableau avec des positions GK/DF/MF/FW
    for table in soup.find_all("table"):
        text = table.get_text()
        if any(pos in text for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward", "GK", "DF", "MF", "FW"]):
            rows = table.find_all("tr")
            if len(rows) >= 3:
                return table

    # Stratégie 2: chercher après un heading "Current squad" ou "First team squad"
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text(strip=True).lower()
        if any(k in text for k in ["current squad", "first team squad", "players", "squad"]):
            elem = heading.find_next_sibling()
            for _ in range(10):
                if elem is None:
                    break
                if elem.name == "table":
                    return elem
                elem = elem.find_next_sibling()

    return None


def _clean_text(text: str) -> str:
    """Nettoie le texte d'un joueur (enlève références, notes)."""
    # Enlève les références comme [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # Enlève les parenthèses comme (captain), (on loan), etc.
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.strip()


def parse_squad(html: str, club_name: str) -> pd.DataFrame:
    """Parse la page d'un club pour extraire l'effectif."""
    soup = BeautifulSoup(html, "lxml")
    table = _find_squad_table(soup)

    if table is None:
        logger.warning(f"Aucun tableau d'effectif trouvé pour {club_name}")
        return pd.DataFrame()

    rows_data = []
    current_section = ""  # Goalkeepers, Defenders, etc.

    for row in table.find_all("tr"):
        tds = row.find_all(["td", "th"])
        if not tds:
            continue

        texts = [td.get_text(strip=True) for td in tds]

        # Détecte les lignes de section (ex: "Goalkeepers", "Defenders")
        if len(texts) == 1 and texts[0] and texts[0] in [
            "Goalkeepers", "Defenders", "Midfielders", "Forwards",
            "No.", "Pos.", "Nation", "Player",
        ]:
            if texts[0] in ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]:
                current_section = texts[0]
            continue

        # Détecte les headers du tableau
        if any(t in texts for t in ["No.", "Pos.", "Nation", "Player"]):
            continue

        # Parse la ligne joueur
        if len(texts) >= 4:
            # Structure typique: [No., Pos., Nation, Player, ...]
            # Parfois la première colonne est fusionnée (th)
            no_ = _clean_text(texts[0]) if len(texts) > 0 else ""
            pos = _clean_text(texts[1]) if len(texts) > 1 else ""
            nation = _clean_text(texts[2]) if len(texts) > 2 else ""
            player = _clean_text(texts[3]) if len(texts) > 3 else ""

            # Filtre les lignes non-joueurs (couleurs, etc.)
            non_player_keywords = ["colours", "colors", "home", "away", "third", "kit", "pattern"]
            if any(kw in player.lower() for kw in non_player_keywords):
                continue
            if any(kw in pos.lower() for kw in non_player_keywords):
                continue

            # Détection capitaine
            is_captain = "captain" in " ".join(texts).lower()

            # Déduire la position si manquante
            if not pos and current_section:
                pos_map = {
                    "Goalkeepers": "GK",
                    "Defenders": "DF",
                    "Midfielders": "MF",
                    "Forwards": "FW",
                }
                pos = pos_map.get(current_section, "")

            if player and len(player) > 1:
                rows_data.append({
                    "Club": club_name,
                    "Number": no_,
                    "Position": pos,
                    "Nationality": nation,
                    "Player": player,
                    "Captain": is_captain,
                })

    df = pd.DataFrame(rows_data)
    logger.info(f"{len(df)} joueur(s) extraits pour {club_name}.")
    return df


def scrape_all_squads(output_dir: Path) -> list[Path]:
    """
    Scrape les effectifs de tous les clubs de la Botola Pro.

    Args:
        output_dir: Dossier de sortie pour les CSV

    Returns:
        Liste des fichiers CSV générés
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== ÉTAPE 1: Récupération de la liste des clubs ===")
    season_html = fetch_url(SEASON_URL)
    if not season_html:
        logger.error("Impossible de récupérer la page saison.")
        return []

    clubs = extract_clubs_from_season_page(season_html)
    if not clubs:
        logger.error("Aucun club trouvé.")
        return []

    generated_files = []
    all_players = []

    logger.info("=== ÉTAPE 2: Extraction des effectifs ===")
    for idx, club in enumerate(clubs, 1):
        club_name = club["name"]
        wiki_url = club["wiki_url"]

        if not wiki_url:
            logger.warning(f"[{idx}/{len(clubs)}] {club_name}: pas d'URL Wikipedia.")
            continue

        slug = CLUB_SLUG_MAP.get(club_name, club_name.lower().replace(" ", "-").replace(".", ""))
        out_path = output_dir / f"{slug}.csv"

        if out_path.exists():
            logger.info(f"[{idx}/{len(clubs)}] {club_name}: déjà scrapé, ignoré.")
            df = pd.read_csv(out_path)
            all_players.append(df)
            generated_files.append(out_path)
            continue

        logger.info(f"[{idx}/{len(clubs)}] {club_name} -> {wiki_url}")
        club_html = fetch_url(wiki_url)
        time.sleep(REQUEST_DELAY)

        if not club_html:
            logger.warning(f"  {club_name}: échec récupération page.")
            continue

        df = parse_squad(club_html, club_name)
        if not df.empty:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"  -> {out_path.name} ({len(df)} joueurs)")
            generated_files.append(out_path)
            all_players.append(df)
        else:
            logger.warning(f"  {club_name}: aucun joueur trouvé.")

    # Fichier consolidé
    if all_players:
        combined = pd.concat(all_players, ignore_index=True)
        combined_path = output_dir / "all_players.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        logger.info(f"\nFichier consolidé: {combined_path} ({len(combined)} joueurs au total)")
        generated_files.append(combined_path)

    return generated_files


if __name__ == "__main__":
    out = Path("data/squads_wiki")
    files = scrape_all_squads(out)
    print(f"\n{'='*60}")
    print(f"SCRAPING TERMINÉ")
    print(f"Fichiers générés: {len(files)}")
    print(f"Dossier: {out.resolve()}")
    print(f"{'='*60}")
