"""Bridge entre le scraper FootyStats et le pipeline RAG.

Convertit les CSV produits par footystats_scraper en documents texte
exploitables par LlamaIndex pour l'analyse multi-agents WAC.
"""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _slug_to_name(slug: str) -> str:
    """Convertit un slug de club en nom lisible."""
    # Ex: wydad-athletic-club-2530 -> Wydad Athletic Club
    parts = slug.replace(".csv", "").split("-")
    # Retire le dernier element si c'est un nombre (ID footystats)
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    name = " ".join(parts).title()
    # Corrections manuelles pour les clubs marocains connus
    corrections = {
        "Wydad Athletic Club": "Wydad AC",
        "Raja Club Athletic De Casablanca": "Raja CA",
        "As Forces Armees Royales De Rabat": "FAR Rabat",
        "Fath Union Sport De Rabat": "FUS Rabat",
        "Ittihad Riadhi De Tanger": "Ittihad Tanger",
        "Olympique Club De Safi": "Olympic Safi",
        "Hassania Union Sport Dagadir": "Hassania Agadir",
        "Maghreb As De Fes": "Maghreb Fes",
        "Kawkab Athletique Club De Marrakech": "Kawkab Marrakech",
        "Renaissance Sportive De Berkane": "RSB Berkane",
        "Difaa Hassani Del Jadida": "DH El Jadida",
        "Club Omnisports De Meknes": "CODM Meknes",
        "Us Yacoub El Mansour": "Yacoub El Mansour",
        "Uts Rabat": "UTS Rabat",
        "Olympique Dcheira": "Olympique Dcheira",
        "Club Renaissance Khemis Zemamra": "CR Khemis Zemamra",
    }
    return corrections.get(name, name)


def convert_team_stats(csv_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Convertit un fichier CSV de stats d'equipe en document texte.

    Returns:
        Chemin du fichier texte genere, ou None si erreur.
    """
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
        f"ATTENTION : Ce document contient les statistiques COLLECTIVES de l'equipe",
        f"'{team_name}' dans la Botola Pro 2025/2026. Ces chiffres representent",
        f"les performances de l'EQUIPE ENTIERE, pas d'un joueur individuel.",
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


def convert_all_team_stats(
    teams_csv_dir: Path, output_dir: Path
) -> List[Path]:
    """
    Convertit tous les CSV d'equipes en documents texte.

    Args:
        teams_csv_dir: Dossier contenant les CSV d'equipes (ex: data/teams)
        output_dir: Dossier de sortie pour les .txt

    Returns:
        Liste des fichiers texte generes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    if not teams_csv_dir.exists():
        print(f"[Bridge] Dossier non trouve : {teams_csv_dir}")
        return generated

    for csv_file in sorted(teams_csv_dir.glob("*.csv")):
        out = convert_team_stats(csv_file, output_dir)
        if out:
            generated.append(out)
            print(f"[Bridge] {out.name} genere ({out.stat().st_size} octets)")

    print(f"[Bridge] {len(generated)} fiches equipe generees.")
    return generated


def convert_fixtures(
    fixtures_csv: Path, output_dir: Path, team_filter: Optional[str] = None
) -> Optional[Path]:
    """
    Convertit fixtures.csv en resume de matchs texte.

    Args:
        fixtures_csv: Chemin vers fixtures.csv
        output_dir: Dossier de sortie
        team_filter: Si specifie, ne garde que les matchs impliquant cette equipe

    Returns:
        Chemin du fichier texte genere.
    """
    try:
        df = pd.read_csv(fixtures_csv)
    except Exception as exc:
        print(f"[Bridge] Erreur lecture fixtures : {exc}")
        return None

    if team_filter:
        df = df[
            df["Home"].str.contains(team_filter, case=False, na=False)
            | df["Away"].str.contains(team_filter, case=False, na=False)
        ]
        title = f"CALENDRIER ET RESULTATS - {team_filter}"
        fname = "fixtures_wac.txt"
    else:
        title = "CALENDRIER ET RESULTATS - BOTOLA PRO 2025/2026"
        fname = "fixtures_all.txt"

    lines = [title, "=" * 50, ""]

    # Trier par date si possible
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
            lines.append(f"[{date}] {home} vs {away} - A venir")

    text = "\n".join(lines)

    out_path = output_dir / fname
    out_path.write_text(text, encoding="utf-8")
    print(f"[Bridge] {out_path.name} genere ({len(df)} matchs)")
    return out_path


def convert_players(players_csv: Path, output_dir: Path) -> Optional[Path]:
    """
    Convertit players.csv en fiches joueurs texte.

    Returns:
        Chemin du fichier texte genere.
    """
    try:
        df = pd.read_csv(players_csv)
    except Exception as exc:
        print(f"[Bridge] Erreur lecture players : {exc}")
        return None

    lines = [
        "CLASSEMENTS JOUEURS - BOTOLA PRO 2025/2026",
        "=" * 50,
        "",
        "ATTENTION IMPORTANT :",
        "Ce document contient des classements GLOBAUX de la Botola Pro.",
        "Les joueurs listes ci-dessous ne sont PAS associes a une equipe",
        "specifique dans cette base de donnees. Il est IMPOSSIBLE de savoir",
        "quel joueur joue pour quelle equipe a partir de ce fichier seul.",
        "Ne faites JAMAIS l'hypothese qu'un joueur appartient au WAC ou a",
        "un autre club sans preuve explicite dans les donnees.",
        "",
    ]

    current_category = ""
    for _, row in df.iterrows():
        category = str(row.get("Category", "Unknown")).strip()
        rank = str(row.get("Rank", "")).strip()
        player = str(row.get("Player", "")).strip()
        value = str(row.get("Value", "")).strip()
        metric = str(row.get("Metric", "")).strip()

        if not player:
            continue

        if category != current_category and category != "Unknown":
            lines.append(f"\n## {category}")
            current_category = category

        lines.append(f"{rank}. {player} - {value} {metric}")

    text = "\n".join(lines)

    out_path = output_dir / "players_leaderboard.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[Bridge] {out_path.name} genere")
    return out_path


def run_footystats_bridge(
    scraper_data_dir: Path,
    rag_raw_dir: Path,
    team_focus: str = "Wydad Casablanca",
) -> List[Path]:
    """
    Pipeline complet : convertit toutes les donnees FootyStats en texte pour le RAG.

    Args:
        scraper_data_dir: Dossier racine des donnees scrappees (ex: ScrappingDataBotola/data)
        rag_raw_dir: Dossier data/raw du projet RAG
        team_focus: Equipe a mettre en avant dans les filtres

    Returns:
        Liste de tous les fichiers texte generes.
    """
    print("\n" + "=" * 60)
    print("CONVERSION FOOTYSTATS -> RAG")
    print("=" * 60 + "\n")

    generated: List[Path] = []

    # 1. Stats par equipe
    teams_dir = scraper_data_dir / "teams"
    generated.extend(convert_all_team_stats(teams_dir, rag_raw_dir))

    # 2. Fixtures (tous + filtre WAC)
    fixtures_csv = scraper_data_dir / "fixtures.csv"
    if fixtures_csv.exists():
        out = convert_fixtures(fixtures_csv, rag_raw_dir)
        if out:
            generated.append(out)
        out_wac = convert_fixtures(fixtures_csv, rag_raw_dir, team_filter=team_focus)
        if out_wac:
            generated.append(out_wac)

    # 3. Players
    players_csv = scraper_data_dir / "players.csv"
    if players_csv.exists():
        out = convert_players(players_csv, rag_raw_dir)
        if out:
            generated.append(out)

    print(f"\n[Bridge] Total fichiers generes : {len(generated)}")
    print(f"[Bridge] Destination : {rag_raw_dir.resolve()}")
    return generated
