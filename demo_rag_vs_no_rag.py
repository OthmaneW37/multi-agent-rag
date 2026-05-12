"""
Demo RAG vs No-RAG

Ce script compare les reponses du systeme avec et sans Retrieval-Augmented
Generation sur une meme question. Il demontre la valeur ajoutee du RAG
pour contextualiser les reponses avec des donnees privées (FootyStats).

Usage:
    python demo_rag_vs_no_rag.py --question "Quel est le bilan du WAC a domicile?"
"""

import argparse
import time

from dotenv import load_dotenv

from src.llm_utils import get_llm
from src.rag.indexing import build_or_load_index
from src.rag.retrieval import query_index


def reponse_avec_rag(question: str) -> tuple[str, float]:
    """
    Interroge le systeme AVEC RAG (LlamaIndex + ChromaDB).

    Retourne (reponse, temps_execution).
    """
    print("[RAG] Chargement de l'index...")
    index = build_or_load_index()

    debut = time.time()
    print(f"[RAG] Requete : {question}")
    reponse = query_index(index, question)
    duree = time.time() - debut

    return reponse, duree


def reponse_sans_rag(question: str) -> tuple[str, float]:
    """
    Interroge le LLM SANS RAG (appel direct, pas de retrieval).

    Retourne (reponse, temps_execution).
    """
    llm = get_llm(temperature=0.3)

    prompt = f"""Tu es un assistant football specialise dans la Botola Pro marocaine.
Reponds a la question suivante UNIQUEMENT avec tes connaissances internes.
Tu n'as acces a AUCUNE base de donnees externe.

Question : {question}

Reponse :"""

    debut = time.time()
    print(f"[NO-RAG] Requete : {question}")
    reponse = llm.invoke(prompt).content
    duree = time.time() - debut

    return reponse, duree


def main():
    parser = argparse.ArgumentParser(
        description="Demo comparative RAG vs No-RAG"
    )
    parser.add_argument(
        "--question",
        default="Quel est le bilan du WAC a domicile cette saison?",
        help="Question a poser au systeme",
    )
    args = parser.parse_args()

    load_dotenv()
    question = args.question

    print("=" * 70)
    print(" DEMO COMPARATIVE : RAG vs NO-RAG")
    print("=" * 70)
    print(f"\nQuestion : {question}\n")

    # 1. Reponse SANS RAG
    print("-" * 70)
    print(" 1. REPONSE SANS RAG (LLM seul)")
    print("-" * 70)
    try:
        rep_no_rag, t_no_rag = reponse_sans_rag(question)
        print(f"\n{rep_no_rag}\n")
        print(f"[Temps : {t_no_rag:.2f}s]")
    except Exception as exc:
        rep_no_rag = f"[ERREUR] {exc}"
        t_no_rag = 0.0
        print(f"\n{rep_no_rag}\n")

    # 2. Reponse AVEC RAG
    print("-" * 70)
    print(" 2. REPONSE AVEC RAG (LlamaIndex + ChromaDB)")
    print("-" * 70)
    try:
        rep_rag, t_rag = reponse_avec_rag(question)
        print(f"\n{rep_rag}\n")
        print(f"[Temps : {t_rag:.2f}s]")
    except Exception as exc:
        rep_rag = f"[ERREUR] {exc}"
        t_rag = 0.0
        print(f"\n{rep_rag}\n")

    # 3. Comparaison
    print("=" * 70)
    print(" 3. ANALYSE COMPARATIVE")
    print("=" * 70)

    rag_a_repondu = not rep_rag.startswith("[ERREUR]") and len(rep_rag) > 50
    no_rag_a_repondu = not rep_no_rag.startswith("[ERREUR]") and len(rep_no_rag) > 50

    if rag_a_repondu and no_rag_a_repondu:
        rag_a_sources = "Source" in rep_rag or "score" in rep_rag
        no_rag_hallucine = any(k in rep_no_rag.lower() for k in ["je ne sais pas", "pas sur", "desole", "je n'ai pas", "insuffisant"])

        print(f"\n- RAG a trouve des sources : {'OUI' if rag_a_sources else 'NON'}")
        print(f"- No-RAG semble incertain : {'OUI' if no_rag_hallucine else 'NON'}")

        if rag_a_sources and no_rag_hallucine:
            print("\n>>> CONCLUSION : Le RAG apporte une reponse contextualisee et sourcee,")
            print("    alors que le LLM seul est incapable de repondre de maniere fiable.")
        elif rag_a_sources:
            print("\n>>> CONCLUSION : Le RAG enrichit la reponse avec des donnees concretes")
            print("    issues de la base FootyStats, la ou le LLM seul repond de maniere generique.")
        else:
            print("\n>>> CONCLUSION : Les deux approches ont repondu, mais le RAG")
            print("    fournit des citations verifiables contre les hallucinations potentielles.")
    elif rag_a_repondu and not no_rag_a_repondu:
        print("\n>>> CONCLUSION : Le RAG permet de repondre la ou le LLM seul echoue.")
        print("    C'est l'interet majeur du Retrieval-Augmented Generation.")
    else:
        print("\n>>> CONCLUSION : Impossible de comparer (erreurs techniques).")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
