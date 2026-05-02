# -*- coding: utf-8 -*-
"""
footystats_scraper/scraper.py
-----------------------------
Scraper complet pour la Botola Pro (FootyStats.org).

Livrables :
    1. data/teams/{club_slug}.csv  -> 1 fichier par équipe
    2. data/fixtures.csv           -> Confrontations directes
    3. data/players.csv            -> Fiches joueurs

Usage direct :
    python scraper.py

Usage depuis un autre script :
    from scraper import FootyStatsScraper
    scraper = FootyStatsScraper(output_dir="mon_projet/data")
    scraper.run_all()
"""

import logging
import re
import time
from io import StringIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LEAGUE_URL = "https://footystats.org/morocco/botola-pro"
FIXTURES_URL = "https://footystats.org/morocco/botola-pro/fixtures"
PLAYERS_URL = "https://footystats.org/morocco/botola-pro/players"

COOLDOWN = 4
CHROME_VERSION = 147

CLOUDFLARE_MARKERS = [
    "challenges.cloudflare.com", "cf-challenge", "Just a moment",
    "Please wait", "Checking your browser", "DDoS protection by Cloudflare",
    "Enable JavaScript and cookies to continue", "challenge-error-text",
    "challenge-form", "Turnstile", "Un instant", "cf-turnstile",
]


def is_cloudflare_challenge(html: str) -> bool:
    return any(m.lower() in html.lower() for m in CLOUDFLARE_MARKERS)


def ensure_dirs(base_dir: Path) -> None:
    (base_dir / "teams").mkdir(parents=True, exist_ok=True)
    logger.info(f"Dossiers prêts : {base_dir.resolve()}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def setup_driver(headless: bool = False) -> uc.Chrome:
    logger.info("Lancement de Chrome (undetected)...")
    options = uc.ChromeOptions()
    if not headless:
        options.add_argument("--start-maximized")
    return uc.Chrome(options=options, headless=headless, version_main=CHROME_VERSION)


# ---------------------------------------------------------------------------
# Navigation + Cloudflare
# ---------------------------------------------------------------------------
def fetch_page(driver: uc.Chrome, url: str, timeout: int = 300) -> str:
    logger.info(f"Navigation -> {url}")
    driver.get(url)

    elapsed = 0
    poll = 3
    while elapsed < timeout:
        html = driver.page_source
        if not is_cloudflare_challenge(html):
            break
        if elapsed == 0:
            logger.warning("Cloudflare détecté. Cliquez sur le Captcha si besoin...")
        time.sleep(poll)
        elapsed += poll
    else:
        raise RuntimeError(f"Cloudflare non résolu après {timeout}s.")

    logger.info(f"Cloudflare passé en {elapsed}s.")
    time.sleep(5)
    return driver.page_source


# ---------------------------------------------------------------------------
# Parsing Pandas + nettoyage
# ---------------------------------------------------------------------------
def parse_tables(html: str) -> list[pd.DataFrame]:
    tables = pd.read_html(StringIO(html), flavor="lxml")
    cleaned = []
    for df in tables:
        if df.empty:
            continue
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", regex=True, na=False)]
        df = df.dropna(axis=1, how="all")
        if not df.empty and len(df.columns) >= 2:
            cleaned.append(df)
    logger.info(f"{len(cleaned)} tableau(x) pertinent(s) sur {len(tables)}.")
    return cleaned


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  -> {path.name} ({len(df)} lignes, {len(df.columns)} cols)")


# ---------------------------------------------------------------------------
# 1. Extraction des liens clubs depuis la page league
# ---------------------------------------------------------------------------
def extract_club_links(html: str, base_url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    clubs: dict[str, str] = {}

    for a in soup.find_all("a", href=re.compile(r"^/clubs/[^/]+$")):
        href = a.get("href", "")
        if not href:
            continue

        img = a.find("img")
        name = img.get("alt", "").strip() if img else ""
        if not name:
            name = a.get_text(strip=True).split("\n")[0].strip()
        if not name or len(name) < 2:
            name = href.strip("/").split("/")[-1]

        name = re.sub(r"\s+Stats$", "", name, flags=re.IGNORECASE)

        if re.match(r"^\d+(\.\d+)?$", name) or len(name) > 80:
            continue

        full = urljoin(base_url, href)
        if full not in clubs.values():
            clubs[name] = full

    logger.info(f"{len(clubs)} club(s) trouvé(s) sur la page league.")
    return clubs


# ---------------------------------------------------------------------------
# 2. Scrape une page et sauvegarde le tableau principal
# ---------------------------------------------------------------------------
def is_team_stats_table(df: pd.DataFrame) -> bool:
    cols = [str(c).lower() for c in df.columns]
    return any("stats" in c for c in cols)


def scrape_generic(driver: uc.Chrome, url: str, out_path: Path, filter_fn=None) -> None:
    html = fetch_page(driver, url)
    tables = parse_tables(html)
    if not tables:
        logger.warning(f"Aucun tableau trouvé pour {url}")
        return

    if filter_fn:
        candidates = [t for t in tables if filter_fn(t)]
        if candidates:
            main_df = candidates[0]
        else:
            logger.warning(f"Aucun tableau ne correspond au filtre pour {url}")
            return
    else:
        main_df = max(tables, key=lambda d: len(d) * len(d.columns))

    save_csv(main_df, out_path)


# ---------------------------------------------------------------------------
# 3. Parser spécifique pour la page Fixtures
# ---------------------------------------------------------------------------
def parse_fixtures(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    for match in soup.find_all("ul", class_=re.compile(r"match\s+row\s+cf")):
        date_li = match.find("li", class_="date")
        date = ""
        if date_li:
            span = date_li.find("span", class_="timezone-convert-match-month")
            date = span.get_text(strip=True) if span else ""

        home_a = match.find("a", class_=re.compile(r"team\s+home"))
        away_a = match.find("a", class_=re.compile(r"team\s+away"))
        home_span = home_a.find("span", class_=re.compile(r"hover-modal")) if home_a else None
        away_span = away_a.find("span", class_=re.compile(r"hover-modal")) if away_a else None
        home_team = home_span.get_text(strip=True) if home_span else ""
        away_team = away_span.get_text(strip=True) if away_span else ""

        score_span = match.find("span", class_="ft-score")
        score = score_span.get_text(strip=True) if score_span else ""

        status_span = match.find("span", class_="ft-indicator")
        status = status_span.get_text(strip=True) if status_span else ""

        rows.append({
            "Date": date,
            "Home": home_team,
            "Away": away_team,
            "Score": score,
            "Status": status,
        })

    df = pd.DataFrame(rows)
    logger.info(f"{len(df)} match(s) extraits de la page Fixtures.")
    return df


# ---------------------------------------------------------------------------
# 4. Parser spécifique pour la page Players
# ---------------------------------------------------------------------------
def parse_players(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    for ul in soup.find_all("ul", class_="ui-list"):
        title_li = ul.find("li", class_="list-title")
        category = "Unknown"
        if title_li:
            h2 = title_li.find("h2")
            if h2:
                category = h2.get_text(strip=True)

        for li in ul.find_all("li", class_="list-row"):
            rank_span = li.find("span", class_="rank")
            rank = rank_span.get_text(strip=True) if rank_span else ""

            a = li.find("a")
            name = a.get_text(strip=True) if a else ""

            bubble = li.find("div", class_="small-bubble")
            stat_val = ""
            stat_label = ""
            if bubble:
                stat_val = bubble.contents[0].strip() if bubble.contents else ""
                span = bubble.find("span")
                stat_label = span.get_text(strip=True) if span else ""

            rows.append({
                "Category": category,
                "Rank": rank,
                "Player": name,
                "Value": stat_val,
                "Metric": stat_label,
            })

    df = pd.DataFrame(rows)
    df = df[df["Rank"].astype(str).str.strip() != ""]
    df = df[df["Player"].astype(str).str.strip() != ""]
    logger.info(f"{len(df)} joueur(s) extraits de la page Players.")
    return df


# ---------------------------------------------------------------------------
# 4b. Parser specifique pour la page Squad d'un club
# ---------------------------------------------------------------------------
def parse_squad(html: str, club_name: str) -> pd.DataFrame:
    """
    Parse la page squad d'un club pour extraire la liste des joueurs.

    NOTE: FootyStats gratuit peut avoir des limitations sur les squads.
    Cette fonction essaie plusieurs strategies de parsing.
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    # Strategie 1: Tableaux HTML standards
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers:
            continue

        # Cherche des headers typiques de squad (Name, Position, Age, etc.)
        squad_markers = ["name", "player", "position", "age", "nat", "number", "apps"]
        if any(marker in " ".join(headers).lower() for marker in squad_markers):
            for tr in table.find_all("tr")[1:]:  # Skip header
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    row_data = {"Club": club_name}
                    for i, td in enumerate(tds):
                        if i < len(headers):
                            row_data[headers[i]] = td.get_text(strip=True)
                        else:
                            row_data[f"Col_{i}"] = td.get_text(strip=True)
                    rows.append(row_data)

    # Strategie 2: Listes ul/li
    if not rows:
        for ul in soup.find_all("ul", class_=re.compile(r"player|squad|team")):
            for li in ul.find_all("li"):
                name_tag = li.find("span", class_=re.compile(r"name|player"))
                pos_tag = li.find("span", class_=re.compile(r"position|pos"))
                age_tag = li.find("span", class_=re.compile(r"age"))

                name = name_tag.get_text(strip=True) if name_tag else li.get_text(strip=True)
                if name and len(name) > 2:
                    rows.append({
                        "Club": club_name,
                        "Name": name,
                        "Position": pos_tag.get_text(strip=True) if pos_tag else "",
                        "Age": age_tag.get_text(strip=True) if age_tag else "",
                    })

    df = pd.DataFrame(rows)
    logger.info(f"{len(df)} joueur(s) extraits du squad de {club_name}.")
    return df


# ---------------------------------------------------------------------------
# 5. Classe principale (pour usage externe)
# ---------------------------------------------------------------------------
class FootyStatsScraper:
    """
    Scraper FootyStats pour la Botola Pro.

    Args:
        output_dir: Dossier de sortie pour les CSV (défaut: ./data).
        headless: Lance Chrome en arrière-plan (déconseillé avec Cloudflare).
    """

    def __init__(self, output_dir: str = "./data", headless: bool = False):
        self.output_dir = Path(output_dir)
        self.teams_dir = self.output_dir / "teams"
        self.headless = headless
        self.driver = None

    def start(self) -> None:
        ensure_dirs(self.output_dir)
        self.driver = setup_driver(headless=self.headless)

    def stop(self) -> None:
        if self.driver:
            self.driver.quit()
            logger.info("Navigateur fermé.")

    def scrape_teams(self) -> None:
        if not self.driver:
            raise RuntimeError("Appelle start() avant de scraper.")

        logger.info("=== ÉTAPE 1 : Récupération de la liste des clubs ===")
        league_html = fetch_page(self.driver, LEAGUE_URL)
        clubs = extract_club_links(league_html, LEAGUE_URL)

        if not clubs:
            logger.error("Aucun club trouvé.")
            return

        logger.info("=== ÉTAPE 2 : Extraction des stats par club ===")
        for idx, (name, url) in enumerate(clubs.items(), 1):
            slug = urlparse(url).path.strip("/").split("/")[-1]
            out = self.teams_dir / f"{slug}.csv"
            if out.exists():
                logger.info(f"[{idx}/{len(clubs)}] {name} déjà scrapé, ignoré.")
                continue
            try:
                logger.info(f"[{idx}/{len(clubs)}] {name}")
                scrape_generic(self.driver, url, out, filter_fn=is_team_stats_table)
                time.sleep(COOLDOWN)
            except Exception as exc:
                logger.error(f"Échec {name} : {exc}")
                continue

    def scrape_fixtures(self) -> None:
        if not self.driver:
            raise RuntimeError("Appelle start() avant de scraper.")

        logger.info("=== ÉTAPE 3 : Extraction des confrontations ===")
        try:
            html = fetch_page(self.driver, FIXTURES_URL)
            df = parse_fixtures(html)
            if not df.empty:
                save_csv(df, self.output_dir / "fixtures.csv")
            time.sleep(COOLDOWN)
        except Exception as exc:
            logger.error(f"Échec fixtures : {exc}")

    def scrape_players(self) -> None:
        if not self.driver:
            raise RuntimeError("Appelle start() avant de scraper.")

        logger.info("=== ÉTAPE 4 : Extraction des fiches joueurs ===")
        try:
            html = fetch_page(self.driver, PLAYERS_URL)
            df = parse_players(html)
            if not df.empty:
                save_csv(df, self.output_dir / "players.csv")
            time.sleep(COOLDOWN)
        except Exception as exc:
            logger.error(f"Échec players : {exc}")

    def scrape_squads(self) -> None:
        """
        ÉTAPE 5 : Extraction des squads (effectifs) de chaque club.

        Accède à la page /squad de chaque club et extrait la liste des joueurs.
        NOTE: Cette étape peut échouer si FootyStats bloque l'accès aux squads.
        """
        if not self.driver:
            raise RuntimeError("Appelle start() avant de scraper.")

        logger.info("=== ÉTAPE 5 : Extraction des squads ===")

        # Recharge la liste des clubs depuis la page league
        try:
            league_html = fetch_page(self.driver, LEAGUE_URL)
            clubs = extract_club_links(league_html, LEAGUE_URL)
        except Exception as exc:
            logger.error(f"Impossible de recuperer la liste des clubs : {exc}")
            return

        if not clubs:
            logger.error("Aucun club trouve pour le scraping des squads.")
            return

        squads_dir = self.output_dir / "squads"
        squads_dir.mkdir(parents=True, exist_ok=True)

        for idx, (name, url) in enumerate(clubs.items(), 1):
            slug = urlparse(url).path.strip("/").split("/")[-1]
            squad_url = url + "/squad"
            out = squads_dir / f"{slug}_squad.csv"

            if out.exists():
                logger.info(f"[{idx}/{len(clubs)}] {name} squad deja scrapé, ignore.")
                continue

            try:
                logger.info(f"[{idx}/{len(clubs)}] Squad {name} -> {squad_url}")
                html = fetch_page(self.driver, squad_url)
                df = parse_squad(html, name)
                if not df.empty:
                    save_csv(df, out)
                    logger.info(f"  Squad {name} : {len(df)} joueurs")
                else:
                    logger.warning(f"  Squad {name} : aucun joueur trouve")
                time.sleep(COOLDOWN)
            except Exception as exc:
                logger.error(f"Échec squad {name} : {exc}")
                continue

    def run_all(self) -> None:
        """Exécute les 5 étapes séquentiellement."""
        try:
            self.start()
            self.scrape_teams()
            self.scrape_fixtures()
            self.scrape_players()
            self.scrape_squads()
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=== Démarrage scraper Botola Pro ===")
    scraper = FootyStatsScraper()
    scraper.run_all()
    logger.info("=== Terminé ===")
