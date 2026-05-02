"""Bridge enrichi entre le scraper FootyStats et le pipeline RAG.

Utilise TOUS les tableaux scrapes (table_0 a table_49+) pour creer des
documents texte riches exploitables par le RAG.
"""

import re
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd


# =============================================================================
# AJOUTS MANUELS - INFOS COMPETITION
# =============================================================================
# Ces donnees ne sont pas disponibles sur FootyStats, elles sont ajoutees
# manuellement pour enrichir le rapport.

INFOS_COMPETITION = """
COMPETITION - BOTOLA PRO 2025/2026
================================================================

FORMAT DE LA COMPETITION:
- 16 equipes
- 30 journees (aller-retour)
- 3 points pour une victoire, 1 pour un nul, 0 pour une defaite
- Les 2 derniers sont relegues en Botola 2
- Le champion participe a la Ligue des Champions de la CAF

STADES PRINCIPAUX (approximatifs):
- Wydad AC: Stade Mohammed V, Casablanca (capacite: 67 000)
- Raja CA: Stade Mohammed V, Casablanca (capacite: 67 000)
- FAR Rabat: Stade Moulay Hassan, Rabat (capacite: 12 000)
- FUS Rabat: Stade Belvédère, Rabat (capacite: 15 000)
- Ittihad Tanger: Stade Ibn Batouta, Tanger (capacite: 65 000)
- RSB Berkane: Stade Municipal de Berkane (capacite: 15 000)
- Hassania Agadir: Stade Adrar, Agadir (capacite: 45 000)
- Kawkab Marrakech: Stade de Marrakech (capacite: 45 000)
- Maghreb Fes: Stade Hassan-II, Fes (capacite: 22 000)
- Olympic Safi: Stade El Massira, Safi (capacite: 15 000)
- DH El Jadida: Stade Ben M'Hamed El Abdi, El Jadida (capacite: 15 000)
- CODM Meknes: Stade d'Honneur, Meknes (capacite: 20 000)

REGLES SPECIFIQUES:
- 5 remplacements autorises par match
- VAR utilisee a partir des quarts de finale de la Coupe du Trone
- Limite de 4 joueurs etrangers par equipe

HISTORIQUE RECENT:
- Wydad AC: Champion 2020/2021, 2021/2022
- Raja CA: Champion 2019/2020, 2022/2023
- AS FAR: Champion 2023/2024

ATTENTION: Les stades et informations ci-dessus sont approximatifs et
peuvent changer. Elles sont fournies a titre indicatif uniquement.
"""


# Corrections de noms de clubs
CLUB_NAME_MAP = {
    "wydad-athletic-club-2530": "Wydad AC",
    "raja-club-athletic-de-casablanca-2536": "Raja CA",
    "as-forces-armees-royales-de-rabat-2532": "FAR Rabat",
    "fath-union-sport-de-rabat-2543": "FUS Rabat",
    "ittihad-riadhi-de-tanger-2529": "Ittihad Tanger",
    "olympique-club-de-safi-2534": "Olympic Safi",
    "hassania-union-sport-dagadir-2531": "Hassania Agadir",
    "maghreb-as-de-fes-2547": "Maghreb Fes",
    "kawkab-athletique-club-de-marrakech-2537": "Kawkab Marrakech",
    "renaissance-sportive-de-berkane-2541": "RSB Berkane",
    "difaa-hassani-del-jadida-2538": "DH El Jadida",
    "club-omnisports-de-meknes-2563": "CODM Meknes",
    "us-yacoub-el-mansour-1322394": "Yacoub El Mansour",
    "uts-rabat-17956": "UTS Rabat",
    "olympique-dcheira-2555": "Olympique Dcheira",
    "club-renaissance-khemis-zemamra-10556": "CR Khemis Zemamra",
}


def _slug_to_name(slug: str) -> str:
    """Convertit un slug en nom lisible."""
    return CLUB_NAME_MAP.get(slug, slug)


def _canonical_name(fixture_name: str) -> str:
    """Normalise un nom d'equipe venant de fixtures.csv vers le nom canonique."""
    mapping = {
        "Wydad Casablanca": "Wydad AC",
        "Raja Casablanca": "Raja CA",
        "Difaâ El Jadida": "DH El Jadida",
        "CODM Meknès": "CODM Meknes",
        "Maghreb Fès": "Maghreb Fes",
        "Olympique Dcheïra": "Olympique Dcheira",
    }
    return mapping.get(fixture_name, fixture_name)


def _read_table(csv_path: Path) -> Optional[pd.DataFrame]:
    """Lit un CSV tableau, retourne None si vide/invalide."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty or len(df.columns) < 2:
            return None
        # Supprime colonnes Unnamed vides
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", regex=True, na=False)]
        df = df.dropna(how="all")
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _format_table(df: pd.DataFrame, title: str) -> List[str]:
    """Formate un DataFrame en lignes de texte."""
    lines = [f"\n### {title}", ""]
    for _, row in df.iterrows():
        items = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
        if items:
            lines.append(" | ".join(items))
    lines.append("")
    return lines


def convert_team_enriched(
    base_slug: str, data_dir: Path, output_dir: Path
) -> Optional[Path]:
    """
    Compile TOUS les tableaux d'un club en un document texte enrichi.

    Args:
        base_slug: ex: "wydad-athletic-club-2530"
        data_dir: Dossier racine des donnees scrapes (ex: ScrappingDataBotola/data)
        output_dir: Dossier de sortie

    Returns:
        Chemin du fichier texte genere.
    """
    team_name = _slug_to_name(base_slug)
    pattern = re.compile(re.escape(base_slug) + r"_table_(\d+)\.csv$")

    # Collecte tous les tableaux tries par numero
    tables: Dict[int, Path] = {}
    for csv_file in data_dir.glob(f"{base_slug}_table_*.csv"):
        m = pattern.match(csv_file.name)
        if m:
            tables[int(m.group(1))] = csv_file

    if not tables:
        return None

    lines = [
        f"FICHE COMPLETE - {team_name}",
        "=" * 60,
        f"Source: FootyStats.org - Botola Pro 2025/2026",
        f"Nombre de tableaux de statistiques: {len(tables)}",
        "",
        "ATTENTION: Ce document contient les statistiques COLLECTIVES de l'equipe.",
        "Ces chiffres representent les performances de l'EQUIPE ENTIERE.",
        "Les joueurs individuels ne sont PAS identifies dans ces tableaux.",
        "",
    ]

    # Titres approximatifs pour les tableaux courants
    TABLE_TITLES = {
        0: "Statistiques Principales (Wins, xG, Buts, Possession)",
        1: "Over / BTTS (Over 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, BTTS %)",
        2: "Under (Under 0.5, 1.5, 2.5, 3.5, 4.5, 5.5)",
        3: "Totaux de Saison (Matches Played, Goals Scored/Conceded, Clean Sheets)",
        4: "Match Corners",
        5: "Team Corners",
        6: "Scored 1st/2nd Half (Buts marques 1ere/2eme mi-temps)",
        7: "Scored 1st/2nd Half Details",
        8: "Conceded 1st/2nd Half (Buts encaisses 1ere/2eme mi-temps)",
        9: "Conceded 1st/2nd Half Details",
        10: "Conceded 1st/2nd Half Extended",
        11: "Scored Timing (15-min intervals)",
        12: "Conceded Timing (15-min intervals)",
        13: "Timing Comparison",
        14: "Scored Average Timing",
        15: "BTTS Stats Detail",
        16: "Scored 15-min Intervals",
        17: "Conceded 15-min Intervals",
        18: "Scored Half Comparison",
        19: "Conceded Half Comparison",
        20: "Corners Earned Full Time",
        21: "Corners Earned 1st Half",
        22: "Corners Earned 2nd Half",
        23: "Corners Conceded Full Time",
        24: "Corners Conceded 1st Half",
        25: "1H Match Cards",
        26: "2H Match Cards",
        27: "Match Cards Detail",
        28: "Cards Earned Full Time",
        29: "Cards Earned 1st Half",
        30: "1H/2H Cards Against",
        31: "2H Cards Against",
        32: "Cards Earned 2nd Half",
        33: "Cards Against Full Time",
        34: "Cards Against 1st Half",
        35: "1st Half Corners & Cards",
        36: "2nd Half Corners & Cards",
        37: "Cards Timing",
        38: "Team Shots",
        39: "Team Shots on Target",
        40: "Expected Goals (xG)",
        41: "Fouls Committed",
        42: "Fouls Against",
        43: "Offsides",
        44: "Throw-ins",
        45: "Goal Kicks",
        46: "Free Kicks",
        47: "Match Form / Last 5",
        48: "Scored/Conceded Both Halves",
        49: "Confrontations Directes (Classement & Resultats vs chaque equipe)",
    }

    for num in sorted(tables.keys()):
        df = _read_table(tables[num])
        if df is None:
            continue
        title = TABLE_TITLES.get(num, f"Tableau {num}")
        lines.extend(_format_table(df, title))

    text = "\n".join(lines)
    out_path = output_dir / f"{base_slug}_COMPLETE.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[Enrich] {out_path.name} ({len(tables)} tableaux, {len(text)} caracteres)")
    return out_path


def convert_team_basic(
    csv_path: Path, output_dir: Path
) -> Optional[Path]:
    """Convertit un CSV de stats basiques en document texte (comme l'ancien bridge)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    slug = csv_path.stem
    team_name = _slug_to_name(slug)

    lines = [
        f"STATISTIQUES EQUIPE - {team_name}",
        "=" * 50,
        "",
        f"ATTENTION: Ce document contient les statistiques COLLECTIVES de l'equipe",
        f"'{team_name}' dans la Botola Pro 2025/2026.",
        "",
    ]

    for _, row in df.iterrows():
        stat = str(row.iloc[0])
        overall = str(row.iloc[1]) if len(row) > 1 else ""
        home = str(row.iloc[2]) if len(row) > 2 else ""
        away = str(row.iloc[3]) if len(row) > 3 else ""

        if not overall:
            continue

        lines.append(f"{stat}:")
        lines.append(f"  - Global: {overall}")
        if home:
            lines.append(f"  - Domicile: {home}")
        if away:
            lines.append(f"  - Exterieur: {away}")
        lines.append("")

    text = "\n".join(lines)
    out_path = output_dir / f"{slug}_stats.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def convert_all_teams_enriched(
    data_dir: Path, output_dir: Path
) -> List[Path]:
    """Convertit tous les clubs: fiches enrichies pour ceux qui ont des tableaux, basiques pour les autres."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # 1. Fiches ENRICHIES pour les clubs qui ont des tableaux multiples
    slugs = set()
    for csv_file in data_dir.glob("*_table_*.csv"):
        stem = csv_file.stem
        slug = stem.rsplit("_table_", 1)[0]
        slugs.add(slug)

    for slug in sorted(slugs):
        out = convert_team_enriched(slug, data_dir, output_dir)
        if out:
            generated.append(out)

    # 2. Fiches BASIQUES pour tous les clubs (depuis data/teams/)
    teams_dir = data_dir / "teams"
    if teams_dir.exists():
        for csv_file in sorted(teams_dir.glob("*.csv")):
            slug = csv_file.stem
            # Si ce club a deja une fiche enrichie, on ne genere pas la basique
            if slug not in slugs:
                out = convert_team_basic(csv_file, output_dir)
                if out:
                    generated.append(out)
                    print(f"[Enrich] {out.name} genere (basique)")

    print(f"\n[Enrich] {len(generated)} fiches generees au total.")
    return generated


def _parse_h2h_table(df: pd.DataFrame, focus_team: str) -> List[str]:
    """Parse le tableau 49 de confrontations directes en texte structure."""
    lines = []
    for _, row in df.iterrows():
        # La colonne 0 contient "1. Maghreb Fes"
        team_raw = str(row.iloc[0]) if len(row) > 0 else ""
        result_home = str(row.iloc[1]) if len(row) > 1 else ""
        result_away = str(row.iloc[2]) if len(row) > 2 else ""

        if not team_raw:
            continue

        # Extrait le nom de l'equipe (retire le numero de classement)
        team_match = re.match(r"^\d+\.\s*(.+)$", team_raw.strip())
        opponent = team_match.group(1).strip() if team_match else team_raw.strip()

        lines.append(f"\n--- {focus_team} vs {opponent} ---")

        # Parse le resultat a domicile (H)
        home_match = re.search(r"\(H\)\s*([\d\-]+)", result_home)
        if home_match:
            score = home_match.group(1).strip()
            lines.append(f"Domicile: {focus_team} {score} {opponent}")
        elif "(H)" in result_home:
            date_match = re.search(r"\(H\)\s*([\d\s\w]+)", result_home)
            if date_match:
                lines.append(f"Domicile: A venir ({date_match.group(1).strip()})")

        # Parse le resultat a l'exterieur (A)
        away_match = re.search(r"\(A\)\s*([\d\-]+)", result_away)
        if away_match:
            score = away_match.group(1).strip()
            lines.append(f"Exterieur: {opponent} {score} {focus_team}")
        elif "(A)" in result_away:
            date_match = re.search(r"\(A\)\s*([\d\s\w]+)", result_away)
            if date_match:
                lines.append(f"Exterieur: A venir ({date_match.group(1).strip()})")

        # Si le resultat est dans la colonne 1 (format alternatif)
        if not home_match and not away_match:
            # Essaie de trouver n'importe quel score
            score_match = re.search(r"([\d\-]+)", result_home)
            if score_match and "-" in score_match.group(1):
                lines.append(f"Resultat: {focus_team} {score_match.group(1)} {opponent}")

    return lines


def convert_head_to_head(
    data_dir: Path, output_dir: Path, focus_team: str = "Wydad AC"
) -> Optional[Path]:
    """
    Extrait les confrontations directes depuis le tableau 49 de chaque equipe
    et cree un document centralise.
    """
    lines = [
        f"CONFRONTATIONS DIRECTES - {focus_team}",
        "=" * 60,
        "",
        "Ce document recense tous les resultats des confrontations directes",
        f"de {focus_team} contre chaque equipe de la Botola Pro 2025/2026.",
        "Source: FootyStats.org",
        "",
        "LEGENDE:",
        "- (H) = Match a domicile de WAC",
        "- (A) = Match a l'exterieur de WAC",
        "- Les scores sont affiches sous format 'WAC X-Y Adversaire'",
        "",
    ]

    focus_slug = None
    for slug, name in CLUB_NAME_MAP.items():
        if name == focus_team:
            focus_slug = slug
            break

    if not focus_slug:
        print(f"[Enrich] Equipe {focus_team} non trouvee dans le mapping.")
        return None

    # Lit le tableau 49 du focus team
    h2h_csv = data_dir / f"{focus_slug}_table_49.csv"
    if h2h_csv.exists():
        df = _read_table(h2h_csv)
        if df is not None:
            h2h_lines = _parse_h2h_table(df, focus_team)
            lines.extend(h2h_lines)
            lines.append("")
        else:
            lines.append("[Donnees de confrontations directes non disponibles]")
    else:
        lines.append("[Fichier de confrontations directes non trouve]")

    # Ajoute aussi les matchs depuis fixtures.csv pour completer
    fixtures_csv = data_dir.parent / "data" / "fixtures.csv"
    wac_matches_by_opponent: dict[str, list[str]] = {}
    all_wac_matches: list[str] = []

    if fixtures_csv.exists():
        try:
            fix_df = pd.read_csv(fixtures_csv)
            wac_matches = fix_df[
                fix_df["Home"].str.contains("Wydad", case=False, na=False)
                | fix_df["Away"].str.contains("Wydad", case=False, na=False)
            ]
            if not wac_matches.empty:
                for _, row in wac_matches.iterrows():
                    date = str(row.get("Date", "")).strip()
                    home = str(row.get("Home", "")).strip()
                    away = str(row.get("Away", "")).strip()
                    score = str(row.get("Score", "")).strip()
                    status = str(row.get("Status", "")).strip()

                    if score:
                        match_line = f"[{date}] {home} {score} {away} ({status})"
                    else:
                        match_line = f"[{date}] {home} vs {away} - A venir ({status})"

                    all_wac_matches.append(match_line)

                    # Determine opponent (canonical name)
                    if "Wydad" in home:
                        opponent = _canonical_name(away)
                    else:
                        opponent = _canonical_name(home)

                    if opponent not in wac_matches_by_opponent:
                        wac_matches_by_opponent[opponent] = []
                    wac_matches_by_opponent[opponent].append(match_line)
        except Exception:
            pass

    # Populate per-opponent sections from fixtures data
    lines.append("\n### CONFRONTATIONS PAR ADVERSAIRE (extraites du calendrier)")
    lines.append("")

    for slug, opponent_name in sorted(CLUB_NAME_MAP.items(), key=lambda x: x[1]):
        if opponent_name == focus_team:
            continue

        lines.append(f"\n--- {focus_team} vs {opponent_name} ---")
        matches = wac_matches_by_opponent.get(opponent_name, [])
        if matches:
            for m in matches:
                lines.append(m)
        else:
            lines.append("(Aucun match trouve dans le calendrier)")

    # Overall match list at the bottom
    if all_wac_matches:
        lines.append("\n\n### Tous les matchs de Wydad AC (Calendrier complet)")
        lines.append("")
        for m in all_wac_matches:
            lines.append(m)
        lines.append("")

    text = "\n".join(lines)
    out_path = output_dir / "confrontations_directes_wac.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[Enrich] {out_path.name} genere ({len(text)} caracteres)")
    return out_path


def convert_fixtures_with_details(
    fixtures_csv: Path, output_dir: Path
) -> Optional[Path]:
    """Convertit fixtures.csv en document detaille avec plus de contexte."""
    try:
        df = pd.read_csv(fixtures_csv)
    except Exception as exc:
        print(f"[Enrich] Erreur lecture fixtures : {exc}")
        return None

    lines = [
        "CALENDRIER COMPLET - BOTOLA PRO 2025/2026",
        "=" * 60,
        "",
        f"Nombre total de matchs: {len(df)}",
        "",
        "ATTENTION: Les dates et horaires sont fournis par FootyStats.",
        "Les stades ne sont PAS specifies dans cette base de donnees.",
        "",
    ]

    # Compte les matchs par equipe
    home_counts = df["Home"].value_counts().to_dict()
    away_counts = df["Away"].value_counts().to_dict()

    # Section prioritaire : matchs a venir (sans score = non joues)
    upcoming = df[df["Score"].isna() | (df["Score"].astype(str).str.strip() == "")]
    if not upcoming.empty:
        lines.append("### MATCHS A VENIR / UPCOMING FIXTURES")
        lines.append("")
        lines.append("ATTENTION: Les matchs listes ci-dessous n'ont pas encore ete joues.")
        lines.append("Les dates et horaires peuvent changer.")
        lines.append("")
        for _, row in upcoming.iterrows():
            date = str(row.get("Date", "")).strip()
            home = str(row.get("Home", "")).strip()
            away = str(row.get("Away", "")).strip()
            status = str(row.get("Status", "")).strip()
            lines.append(f"[{date}] {home} vs {away} ({status})")
        lines.append("")

    lines.append("### Nombre de matchs par equipe")
    lines.append("")
    all_teams = set(home_counts.keys()) | set(away_counts.keys())
    for team in sorted(all_teams):
        total = home_counts.get(team, 0) + away_counts.get(team, 0)
        lines.append(f"- {team}: {total} matchs")
    lines.append("")

    lines.append("### Liste complete des matchs")
    lines.append("")

    for _, row in df.iterrows():
        date = str(row.get("Date", "")).strip()
        home = str(row.get("Home", "")).strip()
        away = str(row.get("Away", "")).strip()
        score = str(row.get("Score", "")).strip()
        status = str(row.get("Status", "")).strip()

        if not home or not away:
            continue

        if score:
            lines.append(f"[{date}] {home} {score} {away} ({status})")
        else:
            lines.append(f"[{date}] {home} vs {away} - A venir (journee non specifiee)")

    text = "\n".join(lines)
    out_path = output_dir / "fixtures_complete.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[Enrich] {out_path.name} genere ({len(df)} matchs, {len(upcoming)} a venir)")
    return out_path


def convert_upcoming_focus_team(
    fixtures_csv: Path, output_dir: Path, focus_team: str = "Wydad AC"
) -> Optional[Path]:
    """Cree un fichier dedie aux prochains matchs de l'equipe mise en avant."""
    try:
        df = pd.read_csv(fixtures_csv)
    except Exception as exc:
        print(f"[Enrich] Erreur lecture fixtures : {exc}")
        return None

    # Matchs a venir impliquant le focus team
    upcoming = df[
        ((df["Home"].str.contains("Wydad", case=False, na=False)) |
         (df["Away"].str.contains("Wydad", case=False, na=False))) &
        (df["Score"].isna() | (df["Score"].astype(str).str.strip() == ""))
    ].copy()

    if upcoming.empty:
        print("[Enrich] Aucun match a venir trouve pour Wydad AC")
        return None

    lines = [
        f"PROCHAIN(S) MATCH(S) - {focus_team}",
        "=" * 60,
        "",
        "ATTENTION: Ces matchs n'ont pas encore ete joues.",
        "Les dates et horaires peuvent changer.",
        "",
    ]

    for _, row in upcoming.iterrows():
        date = str(row.get("Date", "")).strip()
        home = str(row.get("Home", "")).strip()
        away = str(row.get("Away", "")).strip()
        status = str(row.get("Status", "")).strip()
        lines.append(f"[{date}] {home} vs {away} ({status})")

    lines.append("")
    lines.append(f"Total matchs a venir: {len(upcoming)}")
    lines.append("")

    text = "\n".join(lines)
    out_path = output_dir / "prochain_match_wac.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[Enrich] {out_path.name} genere ({len(upcoming)} matchs)")
    return out_path


def run_enriched_bridge(
    scraper_data_dir: Path,
    rag_raw_dir: Path,
    focus_team: str = "Wydad AC",
) -> List[Path]:
    """
    Pipeline enrichi complet.

    Args:
        scraper_data_dir: Dossier racine des donnees scrapes (ex: ScrappingDataBotola/data)
        rag_raw_dir: Dossier data/raw du projet RAG
        focus_team: Equipe a mettre en avant

    Returns:
        Liste de tous les fichiers texte generes.
    """
    print("\n" + "=" * 60)
    print("CONVERSION ENRICHIE FOOTYSTATS -> RAG")
    print("=" * 60 + "\n")

    generated: List[Path] = []

    # 1. Fiches completes par equipe (tous les tableaux)
    generated.extend(convert_all_teams_enriched(scraper_data_dir, rag_raw_dir))

    # 2. Confrontations directes
    out = convert_head_to_head(scraper_data_dir, rag_raw_dir, focus_team)
    if out:
        generated.append(out)

    # 3. Calendrier complet + prochains matchs WAC
    fixtures_csv = scraper_data_dir / "fixtures.csv"
    if fixtures_csv.exists():
        out = convert_fixtures_with_details(fixtures_csv, rag_raw_dir)
        if out:
            generated.append(out)
        out_upcoming = convert_upcoming_focus_team(fixtures_csv, rag_raw_dir, focus_team)
        if out_upcoming:
            generated.append(out_upcoming)

    # 4. Squads (si disponibles) - Wikipedia
    squads_wiki_dir = scraper_data_dir / "squads_wiki"
    if squads_wiki_dir.exists():
        squad_files = sorted(squads_wiki_dir.glob("*.csv"))
        # Exclure all_players.csv qui est le fichier consolidé
        squad_files = [f for f in squad_files if f.name != "all_players.csv"]

        if squad_files:
            lines = [
                "SQUADS / EFFECTIFS - BOTOLA PRO 2025/2026",
                "=" * 60,
                "",
                "Source: Wikipedia EN (pages des clubs)",
                "ATTENTION: Les effectifs peuvent evoluer (transferts, prets).",
                "Les donnees ci-dessous sont a titre indicatif.",
                "",
            ]

            for squad_file in squad_files:
                try:
                    df = pd.read_csv(squad_file)
                    if df.empty:
                        continue

                    club = str(df.iloc[0].get("Club", squad_file.stem))
                    lines.append(f"\n--- {club} ---")
                    lines.append(f"Joueurs trouves: {len(df)}")

                    # Group by position
                    for pos in ["GK", "DF", "MF", "FW"]:
                        pos_players = df[df["Position"] == pos]
                        if not pos_players.empty:
                            pos_names = {"GK": "Gardiens", "DF": "Defenseurs", "MF": "Milieux", "FW": "Attaquants"}
                            lines.append(f"\n{pos_names.get(pos, pos)} ({len(pos_players)}):")
                            for _, row in pos_players.iterrows():
                                num = str(row.get("Number", "")).strip()
                                name = str(row.get("Player", "")).strip()
                                nation = str(row.get("Nationality", "")).strip()
                                captain = row.get("Captain", False)

                                info = f"  #{num} {name}" if num else f"  {name}"
                                if nation and nation != "nan":
                                    info += f" ({nation})"
                                if captain:
                                    info += " [CAPITAINE]"
                                lines.append(info)

                except Exception as exc:
                    print(f"[Enrich] Erreur lecture squad {squad_file.name}: {exc}")

            # Fichier par club
            text = "\n".join(lines)
            out = rag_raw_dir / "squads_all.txt"
            out.write_text(text, encoding="utf-8")
            generated.append(out)
            print(f"[Enrich] {out.name} genere ({len(squad_files)} clubs, ~{sum(len(pd.read_csv(f)) for f in squad_files)} joueurs)")

            # Fichier spécial WAC (pour retrieval ciblée)
            wac_squad = squads_wiki_dir / "wydad-athletic-club-2530.csv"
            if wac_squad.exists():
                df_wac = pd.read_csv(wac_squad)
                if not df_wac.empty:
                    wac_lines = [
                        "EFFECTIF WYDAD AC (WAC) - SAISON 2025/2026",
                        "=" * 60,
                        "",
                        "Source: Wikipedia EN",
                        f"Total joueurs: {len(df_wac)}",
                        "",
                    ]
                    for pos in ["GK", "DF", "MF", "FW"]:
                        pos_players = df_wac[df_wac["Position"] == pos]
                        if not pos_players.empty:
                            pos_names = {"GK": "GARDIENS", "DF": "DEFENSEURS", "MF": "MILIEUX", "FW": "ATTAQUANTS"}
                            wac_lines.append(f"\n{pos_names.get(pos, pos)} ({len(pos_players)} joueurs):")
                            for _, row in pos_players.iterrows():
                                num = str(row.get("Number", "")).strip()
                                name = str(row.get("Player", "")).strip()
                                nation = str(row.get("Nationality", "")).strip()
                                captain = row.get("Captain", False)
                                info = f"  #{num} {name}" if num else f"  {name}"
                                if nation and nation != "nan":
                                    info += f" ({nation})"
                                if captain:
                                    info += " [CAPITAINE]"
                                wac_lines.append(info)

                    wac_text = "\n".join(wac_lines)
                    wac_out = rag_raw_dir / "effectif_wac.txt"
                    wac_out.write_text(wac_text, encoding="utf-8")
                    generated.append(wac_out)
                    print(f"[Enrich] {wac_out.name} genere ({len(df_wac)} joueurs)")

    # 4b. Squads FootyStats (fallback si Wikipedia vide)
    squads_dir = scraper_data_dir / "squads"
    if squads_dir.exists() and not squads_wiki_dir.exists():
        squad_files = list(squads_dir.glob("*_squad.csv"))
        if squad_files:
            lines = [
                "SQUADS / EFFECTIFS - BOTOLA PRO 2025/2026",
                "=" * 60,
                "",
                "ATTENTION: Les joueurs listes ci-dessous sont extraits de FootyStats.",
                "Les positions et ages peuvent etre incomplets.",
                "",
            ]
            for squad_file in sorted(squad_files):
                try:
                    df = pd.read_csv(squad_file)
                    if not df.empty:
                        club = str(df.iloc[0].get("Club", squad_file.stem.replace("_squad", "")))
                        lines.append(f"\n--- {club} ---")
                        lines.append(f"Joueurs trouves: {len(df)}")
                        for _, row in df.head(15).iterrows():
                            name = str(row.get("Name", row.get("name", ""))).strip()
                            pos = str(row.get("Position", row.get("position", ""))).strip()
                            age = str(row.get("Age", row.get("age", ""))).strip()
                            if name:
                                info = f"- {name}"
                                if pos:
                                    info += f" ({pos})"
                                if age:
                                    info += f" - {age} ans"
                                lines.append(info)
                        if len(df) > 15:
                            lines.append(f"... et {len(df) - 15} autres joueurs")
                except Exception as exc:
                    print(f"[Enrich] Erreur lecture squad {squad_file.name}: {exc}")

            text = "\n".join(lines)
            out = rag_raw_dir / "squads_all.txt"
            out.write_text(text, encoding="utf-8")
            generated.append(out)
            print(f"[Enrich] {out.name} genere ({len(squad_files)} squads)")

    # 5. Infos competition (manuel)
    out = rag_raw_dir / "infos_competition.txt"
    out.write_text(INFOS_COMPETITION, encoding="utf-8")
    generated.append(out)
    print(f"[Enrich] {out.name} genere (infos manuelles)")

    print(f"\n[Enrich] Total fichiers generes : {len(generated)}")
    print(f"[Enrich] Destination : {rag_raw_dir.resolve()}")
    return generated
