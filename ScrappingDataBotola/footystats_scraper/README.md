# FootyStats Scraper - Botola Pro

Scraper Python pour extraire les statistiques de football depuis [FootyStats.org](https://footystats.org) pour la **Botola Pro** (championnat du Maroc).

## Livrables

Le scraper produit 3 types de fichiers CSV :

1. **`data/teams/{slug}.csv`** — 1 fichier par club (16 fichiers)
   - Stats globales : Wins, Draws, xG, Possession, Clean Sheets, etc.

2. **`data/fixtures.csv`** — Confrontations directes
   - Date, Domicile, Extérieur, Score, Statut (FT ou à venir)

3. **`data/players.csv`** — Fiches joueurs
   - Top Scorers, Assists, Clean Sheets, etc.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Mode script autonome

```bash
python scraper.py
```

Le navigateur Chrome va s'ouvrir. Si Cloudflare affiche un Captcha, clique sur **"Je ne suis pas un robot"**. Le script reprendra automatiquement.

### 2. Mode intégration (depuis ton projet principal)

Copie ce dossier `footystats_scraper/` dans ton projet, puis :

```python
from footystats_scraper.scraper import FootyStatsScraper

scraper = FootyStatsScraper(output_dir="mon_projet/data")
scraper.run_all()  # Lance tout

# Ou étape par étape :
scraper.start()
scraper.scrape_teams()
scraper.scrape_fixtures()
scraper.scrape_players()
scraper.stop()
```

## Configuration

Modifie `CHROME_VERSION` dans `scraper.py` selon ta version de Google Chrome :

```python
CHROME_VERSION = 147  # Adapte selon chrome://version
```

## Structure du dossier

```
footystats_scraper/
├── scraper.py           # Script principal
├── requirements.txt     # Dépendances
├── README.md            # Ce fichier
└── data/                # CSV générés (créé automatiquement)
    ├── teams/
    │   ├── wydad-athletic-club-2530.csv
    │   └── ...
    ├── fixtures.csv
    └── players.csv
```

## Notes

- Le scraper utilise **undetected-chromedriver** pour contourner Cloudflare.
- Si Chrome se met à jour, ajuste `CHROME_VERSION`.
- Les données sont fraîches à chaque exécution.
