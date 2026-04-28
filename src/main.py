"""Point d'entrée principal du système multi-agents."""

import argparse
import sys
import os

# Force UTF-8 for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ajout du répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.orchestration.orchestrator import Orchestrator
from src.rag.indexing import build_or_load_index
from src.llm_utils import get_llm


def setup_index():
    """Construit ou charge l'index RAG."""
    print("\n[INDEX] Construction/Chargement de l'index RAG...")
    index = build_or_load_index()
    print("[OK] Index pret.\n")
    return index


def demo_complete():
    """Démonstration complète du workflow multi-agents."""
    sujet = input("Entrez le sujet de recherche : ").strip()
    if not sujet:
        sujet = "L'impact des Large Language Models sur la recherche scientifique"
        print(f"Sujet par défaut utilisé : {sujet}")

    orchestrator = Orchestrator()
    state = orchestrator.run(sujet)

    print("\n" + "="*60)
    print("RESULTAT FINAL")
    print("="*60)
    print(state.document)
    print("\n" + "="*60)
    print("RAPPORT DE VERIFICATION")
    print("="*60)
    print(state.rapport_verification)


def demo_agent_individuel():
    """Démonstration d'un agent individuel."""
    print("\nAgents disponibles :")
    print("1. Collecteur - Recherche et extraction de passages")
    print("2. Analyste - Analyse des tendances")
    print("3. Rédacteur - Rédaction de synthèses")
    print("4. Vérificateur - Validation qualité")

    choix = input("\nChoisissez un agent (1-4) : ").strip()
    agents_map = {
        "1": "collecteur",
        "2": "analyste",
        "3": "redacteur",
        "4": "verificateur",
    }

    agent_name = agents_map.get(choix)
    if not agent_name:
        print("Choix invalide.")
        return

    query = input("Entrez votre requête pour l'agent : ").strip()
    if not query:
        query = "Recherche sur les transformers en NLP"
        print(f"Requête par défaut : {query}")

    orchestrator = Orchestrator()
    result = orchestrator.run_agent_only(agent_name, query)

    print("\n" + "="*60)
    print(f"RESULTAT DE L'AGENT {agent_name.upper()}")
    print("="*60)
    print(result.get("output", "Pas de résultat"))


def demo_comparaison_rag():
    """Démonstration comparant avec et sans RAG."""
    query = input("Entrez une question : ").strip()
    if not query:
        query = "Quels sont les avantages des transformers en traitement du langage naturel ?"
        print(f"Question par défaut : {query}")

    print("\n" + "="*60)
    print("REPONSE AVEC RAG (LlamaIndex)")
    print("="*60)
    index = build_or_load_index()
    from src.rag.retrieval import query_index
    response_with_rag = query_index(index, query)
    print(response_with_rag)

    print("\n" + "="*60)
    print("REPONSE SANS RAG (LLM seul)")
    print("="*60)
    llm = get_llm()
    response_without_rag = llm.invoke(f"Réponds à cette question : {query}")
    print(response_without_rag.content)

    print("\n" + "="*60)
    print("COMPARAISON")
    print("="*60)
    print("La réponse AVEC RAG utilise les documents privés indexés.")
    print("La réponse SANS RAG ne repose que sur les connaissances du modèle.")


def main():
    parser = argparse.ArgumentParser(
        description="Système Multi-Agents pour Recherche Académique avec RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python -m src.main --setup                    # Construit l'index RAG
  python -m src.main --demo                     # Lance la démo complète
  python -m src.main --agent                    # Tester un agent individuel
  python -m src.main --compare                  # Comparer avec/sans RAG
        """
    )
    parser.add_argument("--setup", action="store_true", help="Construit ou recharge l'index RAG")
    parser.add_argument("--demo", action="store_true", help="Lance la démonstration complète du workflow")
    parser.add_argument("--agent", action="store_true", help="Teste un agent individuellement")
    parser.add_argument("--compare", action="store_true", help="Compare les réponses avec et sans RAG")
    parser.add_argument("--sujet", type=str, help="Sujet de recherche (pour --demo)")

    args = parser.parse_args()

    if args.setup:
        setup_index()
    elif args.demo:
        setup_index()
        if args.sujet:
            orchestrator = Orchestrator()
            state = orchestrator.run(args.sujet)
            print("\n" + "="*60)
            print("DOCUMENT FINAL")
            print("="*60)
            print(state.document)
        else:
            demo_complete()
    elif args.agent:
        setup_index()
        demo_agent_individuel()
    elif args.compare:
        setup_index()
        demo_comparaison_rag()
    else:
        print("\n" + "="*60)
        print("ASSISTANT DE RECHERCHE ACADEMIQUE MULTI-AGENTS")
        print("="*60)
        print("\nCe systeme coordonne 4 agents intelligents pour produire des etats de l'art :")
        print("  1. Collecteur   - Extraction des passages pertinents (RAG)")
        print("  2. Analyste    - Analyse des tendances et lacunes")
        print("  3. Redacteur   - Redaction de la synthese structuree")
        print("  4. Verificateur - Controle qualite et validation")
        print("\nUtilisez --help pour voir les options disponibles.")
        print("\nExemple rapide : python -m src.main --demo\n")


if __name__ == "__main__":
    main()
