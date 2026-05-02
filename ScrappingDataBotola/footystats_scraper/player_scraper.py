# -*- coding: utf-8 -*-
"""
footystats_scraper/player_scraper.py
------------------------------------
Scraper des stats individuelles des joueurs depuis FootyStats.org.

Pour chaque joueur de la Botola Pro, scrape sa page individuelle et extrait :
    - Métadonnées: nom, club, position, nationalité, âge, numéro
    - Stats générales: matchs joués, minutes, matchs titulaire/remplaçant
    - Attaque: buts, xG, npxG, tirs, conversion
    - Passes: assists, xA, passes, passes clés, centres
    - Dribbles: dribbles réussis, hors-jeu
    - Discipline: cartons jaunes/rouges, fautes
    - Défense: buts encaissés, clean sheets, tacles, interceptions

Usage :
    cd ScrappingDataBotola && python -m footystats_scraper.player_scraper

Livrables :
    data/player_stats.csv  -> Toutes les stats individuelles consolidées
"""

import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from scraper import FootyStatsScraper, fetch_page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LEAGUE_PLAYERS_URL = "https://footystats.org/morocco/botola-pro/players"
OUTPUT_DIR = Path("data")
OUTPUT_CSV = OUTPUT_DIR / "player_stats.csv"
PROGRESS_FILE = OUTPUT_DIR / "player_stats_progress.txt"
COOLDOWN = 4  # secondes entre chaque page joueur


def extract_player_links(html: str) -> list[tuple[str, str]]:
    """Extrait tous les liens joueurs depuis la page /players de la league."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if "/players/" in href and href != "/players/" and text and len(text) < 60:
            full = href if href.startswith("http") else "https://footystats.org" + href
            if full not in seen:
                seen.add(full)
                links.append((text, full))

    logger.info(f"{len(links)} liens joueurs trouvés.")
    return links


def parse_player_page(html: str, player_name: str, player_url: str) -> dict:
    """Parse une page joueur et retourne un dict avec toutes les stats."""
    soup = BeautifulSoup(html, "lxml")
    data = {"player_name": player_name, "player_url": player_url}

    # --- Métadonnées depuis le texte de la page ---
    body_text = soup.get_text()

    # Club / Team
    club_match = re.search(r"Team\s*[:\-]\s*([^\n]+)", body_text)
    data["club"] = club_match.group(1).strip() if club_match else ""

    # Position
    pos_match = re.search(r"Position\s*[:\-]\s*([^\n]+)", body_text)
    data["position"] = pos_match.group(1).strip() if pos_match else ""

    # Nationality
    nat_match = re.search(r"Nationality\s*[:\-]\s*([^\n]+)", body_text)
    data["nationality"] = nat_match.group(1).strip() if nat_match else ""

    # Kit Number
    kit_match = re.search(r"Kit Number\s*[:#\-]*\s*(\d+)", body_text)
    data["kit_number"] = kit_match.group(1) if kit_match else ""

    # Age
    age_match = re.search(r"Age\s*[:\-]*\s*(\d+)", body_text)
    data["age"] = age_match.group(1) if age_match else ""

    # Average Rating
    rating_match = re.search(r"Average Rating.*?([0-9.]+)", body_text)
    data["rating"] = rating_match.group(1) if rating_match else ""

    # --- Stats depuis les tableaux ---
    tables = soup.find_all("table")

    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers or len(headers) < 2:
            continue

        # Détecte la catégorie du tableau depuis le premier header
        category = headers[0].lower()

        for row in table.find_all("tr")[1:]:
            tds = row.find_all(["td", "th"])
            if len(tds) < 2:
                continue

            metric = tds[0].get_text(strip=True)
            total = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            per90 = tds[2].get_text(strip=True) if len(tds) > 2 else ""

            # Normalise la clé
            key = metric.lower().replace(" ", "_").replace("/", "_per_").replace("(", "").replace(")", "").replace("'", "").replace("%", "pct")
            key = re.sub(r"_+", "_", key).strip("_")

            # Stocke la valeur
            data[key] = total
            if per90 and per90.lower() not in ["n/a", "", "nan"]:
                data[key + "_per90"] = per90

    return data


def load_progress() -> set[str]:
    """Charge la liste des joueurs déjà scrapés."""
    if not PROGRESS_FILE.exists():
        return set()
    return set(PROGRESS_FILE.read_text(encoding="utf-8").strip().splitlines())


def save_progress(url: str) -> None:
    """Ajoute une URL à la liste des joueurs scrapés."""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def scrape_all_players() -> None:
    """Scrape les stats de tous les joueurs de la Botola Pro."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scraper = FootyStatsScraper(output_dir=str(OUTPUT_DIR))
    scraper.start()

    try:
        # Étape 1: récupérer la liste des joueurs
        logger.info("=== ÉTAPE 1: Récupération de la liste des joueurs ===")
        fetch_page(scraper.driver, LEAGUE_PLAYERS_URL)
        time.sleep(5)
        league_html = scraper.driver.page_source
        player_links = extract_player_links(league_html)

        if not player_links:
            logger.error("Aucun joueur trouvé.")
            return

        # Étape 2: scrape chaque joueur
        logger.info("=== ÉTAPE 2: Extraction des stats individuelles ===")
        already_scraped = load_progress()
        all_data = []

        # Charge les données existantes si le fichier existe
        if OUTPUT_CSV.exists():
            existing_df = pd.read_csv(OUTPUT_CSV)
            all_data = existing_df.to_dict("records")
            logger.info(f"{len(all_data)} joueurs déjà dans le fichier.")

        for idx, (name, url) in enumerate(player_links, 1):
            if url in already_scraped:
                logger.info(f"[{idx}/{len(player_links)}] {name} déjà scrapé, ignoré.")
                continue

            logger.info(f"[{idx}/{len(player_links)}] {name}")
            try:
                fetch_page(scraper.driver, url)
                time.sleep(COOLDOWN)
                html = scraper.driver.page_source

                player_data = parse_player_page(html, name, url)
                all_data.append(player_data)
                save_progress(url)

                # Sauvegarde incrémentale toutes les 10 joueurs
                if len(all_data) % 10 == 0:
                    pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
                    logger.info(f"  -> Sauvegarde incrémentale ({len(all_data)} joueurs)")

            except Exception as exc:
                logger.error(f"  Échec {name}: {exc}")
                continue

        # Sauvegarde finale
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            logger.info(f"\n=== TERMINÉ ===")
            logger.info(f"Fichier: {OUTPUT_CSV.resolve()}")
            logger.info(f"Joueurs: {len(df)}")
            logger.info(f"Colonnes: {len(df.columns)}")
        else:
            logger.warning("Aucune donnée collectée.")

    finally:
        scraper.stop()


if __name__ == "__main__":
    scrape_all_players()
