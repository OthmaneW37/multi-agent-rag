"""Point d'entree principal de l'application WAC Sport Analytics."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.rag.indexing import build_or_load_index
from src.rag.ingestion import load_documents
from src.rag.footystats_bridge import run_footystats_bridge
from src.rag.enriched_bridge import run_enriched_bridge
from src.orchestration.orchestrator import WACOrchestrator


# Chemin par defaut vers les donnees scrappees
DEFAULT_SCRAPER_DIR = "ScrappingDataBotola/data"


def setup_environment():
    """Charge les variables d'environnement depuis .env."""
    load_dotenv()

    required_vars = ["LLM_PROVIDER"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"ATTENTION : Variables d'environnement manquantes : {missing}")
        print("Verifiez votre fichier .env ou creez-le depuis .env.example")


def run_bridge(
    scraper_dir: str = DEFAULT_SCRAPER_DIR,
    raw_dir: str = "data/raw",
    team_focus: str = "Wydad Casablanca",
):
    """
    Convertit les donnees FootyStats scrappees en documents texte pour le RAG.

    Args:
        scraper_dir: Dossier racine des donnees scrappees
        raw_dir: Dossier de sortie pour les fichiers texte
        team_focus: Equipe a mettre en avant

    Returns:
        Liste des fichiers texte generes
    """
    scraper_path = Path(scraper_dir)
    raw_path = Path(raw_dir)

    if not scraper_path.exists():
        print(f"ERREUR : Dossier scraper introuvable : {scraper_dir}")
        print("Lancez d'abord le scraper FootyStats ou verifiez le chemin.")
        return []

    return run_footystats_bridge(scraper_path, raw_path, team_focus)


def run_ingestion(data_dir: str = "data/raw", rebuild: bool = False):
    """
    Execute l'ingestion des documents et la creation de l'index.

    Args:
        data_dir: Repertoire contenant les documents a indexer
        rebuild: Force la reconstruction de l'index (supprime l'ancien)

    Returns:
        True si succes, False sinon
    """
    print(f"\nIngestion des documents depuis : {data_dir}")

    if not os.path.exists(data_dir):
        print(f"ERREUR : Le repertoire {data_dir} n'existe pas.")
        return False

    files = os.listdir(data_dir)
    if not files:
        print(f"ATTENTION : Le repertoire {data_dir} est vide.")
        return False

    print(f"Trouve {len(files)} fichier(s) a indexer.")

    if rebuild:
        import shutil
        from src.config import config
        if os.path.exists(config.vector_store_path):
            shutil.rmtree(config.vector_store_path)
            print("[RAG] Ancien index supprime.")

    index = build_or_load_index(data_dir, force_rebuild=rebuild)
    print("Index cree avec succes.")

    return True


def run_analysis(
    adversaire: str,
    contexte: str = "",
    output_file: str = None,
):
    """
    Execute le pipeline complet d'analyse pour un match.

    Args:
        adversaire: Nom de l'equipe adverse
        contexte: Informations supplementaires
        output_file: Fichier de sortie pour le rapport (optionnel)

    Returns:
        Dictionnaire des resultats
    """
    print(f"\nAnalyse du match WAC vs {adversaire}")
    if contexte:
        print(f"Contexte : {contexte}")

    orchestrator = WACOrchestrator()

    results = orchestrator.analyser_match(
        adversaire=adversaire,
        contexte=contexte,
        verbose=True,
    )

    # Sauvegarde du rapport Markdown
    if output_file:
        rapport = results.get("rapport_final", "")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rapport)
        print(f"\nRapport sauvegarde dans : {output_file}")

    # Sauvegarde JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"data/processed/analyse_{adversaire.replace(' ', '_')}_{timestamp}.json"
    os.makedirs("data/processed", exist_ok=True)

    serializable_results = {
        "adversaire": adversaire,
        "contexte": contexte,
        "timestamp": timestamp,
        "etapes": results.get("etapes", {}),
        "rapport_final": results.get("rapport_final", ""),
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"Resultats sauvegardes dans : {json_file}")

    return results


def main():
    """Point d'entree principal avec gestion des arguments CLI."""
    parser = argparse.ArgumentParser(
        description="WAC Sport Analytics - Analyse multi-agents pour le Wydad AC"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commande a executer")

    # Commande : bridge (FootyStats -> texte)
    bridge_parser = subparsers.add_parser(
        "bridge", help="Convertir les donnees FootyStats scrappees en texte pour le RAG"
    )
    bridge_parser.add_argument(
        "--scraper-dir",
        default=DEFAULT_SCRAPER_DIR,
        help="Dossier racine des donnees scrappees (defaut: ScrappingDataBotola/data)",
    )
    bridge_parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Dossier de sortie pour les fichiers texte (defaut: data/raw)",
    )
    bridge_parser.add_argument(
        "--team-focus",
        default="Wydad Casablanca",
        help="Equipe a mettre en avant (defaut: Wydad Casablanca)",
    )

    # Commande : ingest
    ingest_parser = subparsers.add_parser("ingest", help="Indexer les documents")
    ingest_parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Repertoire des documents a indexer (defaut: data/raw)",
    )
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force la reconstruction complete de l'index",
    )

    # Commande : analyse
    analyse_parser = subparsers.add_parser("analyse", help="Analyser un match")
    analyse_parser.add_argument(
        "--adversaire",
        required=True,
        help="Nom de l'equipe adverse (ex: 'Raja CA', 'FUS Rabat')",
    )
    analyse_parser.add_argument(
        "--contexte",
        default="",
        help="Contexte du match (ex: 'Match a domicile, 15e journee')",
    )
    analyse_parser.add_argument(
        "--output",
        default=None,
        help="Fichier de sortie pour le rapport (defaut: auto)",
    )

    # Commande : pipeline (ingestion + analyse)
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Ingestion + Analyse en une commande"
    )
    pipeline_parser.add_argument(
        "--adversaire", required=True, help="Nom de l'equipe adverse"
    )
    pipeline_parser.add_argument(
        "--contexte", default="", help="Contexte du match"
    )
    pipeline_parser.add_argument(
        "--data-dir", default="data/raw", help="Repertoire des documents a indexer"
    )
    pipeline_parser.add_argument(
        "--output", default=None, help="Fichier de sortie pour le rapport"
    )

    # Commande : bridge-enriched (version enrichie avec tous les tableaux)
    enriched_parser = subparsers.add_parser(
        "bridge-enriched",
        help="Convertir TOUTES les donnees FootyStats (tableaux enrichis) en texte",
    )
    enriched_parser.add_argument(
        "--scraper-dir",
        default=DEFAULT_SCRAPER_DIR,
        help="Dossier racine des donnees scrappees (defaut: ScrappingDataBotola/data)",
    )
    enriched_parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Dossier de sortie pour les fichiers texte (defaut: data/raw)",
    )

    # Commande : full-pipeline (bridge enrichi + ingest + analyse)
    full_parser = subparsers.add_parser(
        "full-pipeline",
        help="Pipeline complet : Bridge FootyStats + Ingestion + Analyse",
    )
    full_parser.add_argument(
        "--adversaire", required=True, help="Nom de l'equipe adverse"
    )
    full_parser.add_argument(
        "--contexte", default="", help="Contexte du match"
    )
    full_parser.add_argument(
        "--scraper-dir",
        default=DEFAULT_SCRAPER_DIR,
        help="Dossier racine des donnees scrappees",
    )
    full_parser.add_argument(
        "--raw-dir", default="data/raw", help="Dossier de sortie des textes"
    )
    full_parser.add_argument(
        "--output", default=None, help="Fichier de sortie pour le rapport"
    )

    args = parser.parse_args()

    # Setup environnement
    setup_environment()

    if args.command == "bridge":
        files = run_bridge(
            scraper_dir=args.scraper_dir,
            raw_dir=args.raw_dir,
            team_focus=args.team_focus,
        )
        sys.exit(0 if files else 1)

    elif args.command == "bridge-enriched":
        # Nettoie data/raw avant de generer les nouveaux fichiers
        raw_path = Path(args.raw_dir)
        if raw_path.exists():
            for old_file in raw_path.glob("*.txt"):
                old_file.unlink()
            print(f"[Nettoyage] Anciens fichiers supprimes de {args.raw_dir}")
        files = run_enriched_bridge(
            scraper_data_dir=Path(args.scraper_dir),
            rag_raw_dir=raw_path,
        )
        sys.exit(0 if files else 1)

    elif args.command == "ingest":
        success = run_ingestion(args.data_dir, rebuild=args.rebuild)
        sys.exit(0 if success else 1)

    elif args.command == "analyse":
        run_analysis(
            adversaire=args.adversaire,
            contexte=args.contexte,
            output_file=args.output,
        )

    elif args.command == "pipeline":
        print("\n" + "=" * 60)
        print("PHASE 1 : INGESTION DES DOCUMENTS")
        print("=" * 60)
        run_ingestion(args.data_dir, rebuild=True)

        print("\n" + "=" * 60)
        print("PHASE 2 : ANALYSE MULTI-AGENTS")
        print("=" * 60)
        run_analysis(
            adversaire=args.adversaire,
            contexte=args.contexte,
            output_file=args.output,
        )

    elif args.command == "full-pipeline":
        # Phase 1 : Bridge enrichi
        print("\n" + "=" * 60)
        print("PHASE 1 : CONVERSION FOOTYSTATS -> TEXTE (ENRICHIE)")
        print("=" * 60)
        # Nettoie data/raw
        raw_path = Path(args.raw_dir)
        if raw_path.exists():
            for old_file in raw_path.glob("*.txt"):
                old_file.unlink()
            print(f"[Nettoyage] Anciens fichiers supprimes de {args.raw_dir}")
        files = run_enriched_bridge(
            scraper_data_dir=Path(args.scraper_dir),
            rag_raw_dir=raw_path,
        )
        if not files:
            print("ERREUR : La conversion a echoue. Arret.")
            sys.exit(1)

        # Phase 2 : Ingestion
        print("\n" + "=" * 60)
        print("PHASE 2 : INGESTION DES DOCUMENTS")
        print("=" * 60)
        run_ingestion(args.raw_dir, rebuild=True)

        # Phase 3 : Analyse
        print("\n" + "=" * 60)
        print("PHASE 3 : ANALYSE MULTI-AGENTS")
        print("=" * 60)
        run_analysis(
            adversaire=args.adversaire,
            contexte=args.contexte,
            output_file=args.output,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
